'''
Entire training pipeline logic.
Created by Basile Van Hoorick, Jun 2022.
'''

from models.hide_seek.tcow.__init__ import *

# Internal imports.
import models.hide_seek.tcow.data.data_kubric as data_kubric
import models.hide_seek.tcow.data.data_utils as data_utils
import models.hide_seek.tcow.loss as loss
import models.hide_seek.tcow.utils.my_utils as my_utils
import models.hide_seek.tcow.seeker.perfect as perfect
import models.hide_seek.tcow.model.stg_functions as stg_functions

class MyTrainPipeline(torch.nn.Module):
    '''
    Wrapper around most of the training iteration such that DataParallel can be leveraged.
    '''

    def __init__(self, train_args, logger, networks, device):
        super().__init__()
        self.train_args = train_args
        self.logger = logger
        self.networks = torch.nn.ModuleDict(networks)
        self.device = device
        self.phase = None  # Assigned only by set_phase().
        self.losses = None  # Instantiated only by set_phase().
        self.stg_threshold = stg_functions.StraightThroughThreshold()
        # self.kubric_generators = [None for _ in range(train_args.batch_size)]
        self.to_tensor = torchvision.transforms.ToTensor()

    def set_phase(self, phase):
        '''
        Must be called when switching between train / validation / test phases.
        '''
        self.phase = phase
        self.losses = loss.MyLosses(self.train_args, self.logger, phase)

        if 'train' in phase:
            self.train()
            for (k, v) in self.networks.items():
                v.train()
            torch.set_grad_enabled(True)

        else:
            self.eval()
            for (k, v) in self.networks.items():
                v.eval()
            torch.set_grad_enabled(False)

        for (k, v) in self.networks.items():
            if k == 'seeker':
                v.set_phase(phase)

    def forward(self, data_retval, cur_step, total_step, epoch, progress, include_loss,
                metrics_only, perfect_baseline, no_pred, cluster_viz=None):
        '''
        Handles one parallel iteration of the training or validation phase.
        Executes the models and calculates the per-example losses.
        This is all done in a parallelized manner to minimize unnecessary communication.
        :param data_retval (dict): Data loader elements.
        :param cur_step (int): Current data loader index.
        :param total_step (int): Cumulative data loader index, including all previous epochs.
        :param epoch (int): Current epoch index (0-based).
        :param progress (float) in [0, 1]: Total progress within the entire training run.
        :param include_loss (bool).
        :param metrics_only (bool).
        :param perfect_baseline (str).
        :param no_pred (bool).
        :return (model_retval, loss_retval).
            model_retval (dict): All output information.
            loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        '''
        # We are using either our own model (timesformer) or AOT baseline.
        # Proceed with source-specific forward logic.
        source_name = data_retval['source_name'][0]
        assert all([x == source_name for x in data_retval['source_name']]), \
            'Cannot mix sources within one batch.'

        if perfect_baseline != 'none':
            model_retval = perfect.run_baseline(
                data_retval, perfect_baseline, self.logger, self.train_args)

        else:
            if source_name == 'kubric':
                model_retval, cluster_viz  = self.forward_kubric(data_retval, no_pred)

            elif source_name == 'ytvos':
                model_retval = self.forward_ytvos(data_retval)

            elif source_name == 'plugin':
                model_retval, cluster_viz = self.forward_plugin(data_retval)
                # model_retval = self.forward_plugin(data_retval)

        if include_loss:
            loss_retval = self.losses.per_example(data_retval, model_retval, progress, metrics_only)
        else:
            loss_retval = None

        return (model_retval, loss_retval, cluster_viz)

    def forward_kubric(self, data_retval, no_pred):
        # NOTE: Both timesformer and AOT are routed through this method now!
        assert self.train_args.which_seeker == 'mask_track_2d'
        assert self.train_args.seeker_input_format == 'rgb'

        within_batch_inds = data_retval['within_batch_idx']
        B = within_batch_inds.shape[0]

        # Retrieve data.
        # NOTE: Any array with a dimension M will typically have leading zeros.
        kubric_retval = data_retval['kubric_retval']
        all_xyz = kubric_retval['pv_xyz_tf']
        # (B, 3, T, Hf, Wf) or (B, 1).
        all_rgb = kubric_retval['pv_rgb_tf']
        # (B, 3, T, Hf, Wf).
        all_segm = kubric_retval['pv_segm_tf']
        # (B, 1, T, Hf, Wf).
        all_div_segm = kubric_retval['pv_div_segm_tf']
        # (B, M, T, Hf, Wf).
        all_xyz = all_xyz.to(self.device)
        all_rgb = all_rgb.to(self.device)
        all_segm = all_segm.to(self.device)
        all_div_segm = all_div_segm.to(self.device)
        inst_count = kubric_retval['pv_inst_count']
        # (B, 1); acts as Qt value per example.
        query_time = kubric_retval['traject_retval_tf']['query_time']
        # (B).
        occl_fracs = kubric_retval['traject_retval_tf']['occl_fracs_tf']
        # (B, M, T, 3) with (f, v, t).
        occl_cont_dag = kubric_retval['traject_retval_tf']['occl_cont_dag_tf']
        # (B, T, M, M, 3) with (c, od, of).
        # NOTE: ^ Based on non-cropped data, so could be inaccurate near edges!
        target_desirability = kubric_retval['traject_retval_tf']['desirability_tf']
        # (B, M, 7).
        scene_dp = data_retval['scene_dp']

        (T, H, W) = all_rgb.shape[-3:]
        assert T == self.train_args.num_frames
        Qs = self.train_args.num_queries  # Selected per example here.

        # Assemble seeker input (which is always a simple copy now).
        seeker_input = all_rgb  # (B, 3, T, Hf, Wf).

        # Sample either random or biased queries.
        sel_query_inds = my_utils.sample_query_inds(
            B, Qs, inst_count, target_desirability, self.train_args, self.device, self.phase)

        # Loop over selected queries and accumulate results & targets.
        all_occl_fracs = []
        all_desirability = []
        all_seeker_query_mask = []
        all_snitch_occl_by_ptr = []
        all_full_occl_cont_id = []
        all_target_mask = []
        all_output_mask = []
        all_target_flags = []
        all_output_flags = []  # Only if timesformer.
        all_aot_loss = []  # Only if AOT.

        for q in range(Qs):

            # Get query info.
            # NOTE: query_idx is still a (B) tensor, so don't forget to select index.
            # query_idx[b] refers directly to the snitch instance ID we are tracking.
            query_idx = sel_query_inds[:, q]  # (B).
            qt_idx = query_time[0].item()

            # Prepare query mask and ground truths.
            (seeker_query_mask, snitch_occl_by_ptr, full_occl_cont_id, target_mask,
                target_flags) = data_utils.fill_kubric_query_target_mask_flags(
                all_segm, all_div_segm, query_idx, qt_idx, occl_fracs, occl_cont_dag, scene_dp,
                self.logger, self.train_args, self.device, self.phase)

            # Sanity checks.
            if not seeker_query_mask.any():
                raise RuntimeError(
                    f'seeker_query_mask all zero? q: {q} query_idx: {query_idx} qt_idx: {qt_idx}')
            if not target_mask.any():
                raise RuntimeError(
                    f'target_mask all zero? q: {q} query_idx: {query_idx} qt_idx: {qt_idx}')

            output_flags = torch.tensor([0.0], device=self.device)
            aot_loss = torch.tensor([0.0], device=self.device)

            # Run seeker to recover hierarchical masks over time.
            if not(no_pred):
                if self.train_args.tracker_arch == 'aot':
                    # Baseline; no hierarchy, no flags.
                    if 'train' in self.phase or 'val' in self.phase:
                        # NOTE: At train time, this is the point which forces us to align query_mask
                        # with target_mask (named snitch_xray_mask here).
                        snitch_xray_mask = target_mask[:, 0:1]  # (B, 1, T, Hf, Wf)
                        (output_mask, aot_loss, cluster_viz) = self.networks['seeker'](
                            seeker_input, snitch_xray_mask)  # (B, 1, T, Hf, Wf), (1, 3).
                    else:
                        output_mask, cluster_viz = self.networks['seeker'](seeker_input, seeker_query_mask)
                        # (B, 1, T, Hf, Wf).

                else:
                    (output_mask, output_flags, cluster_viz) = self.networks['seeker'](
                        seeker_input, seeker_query_mask)  # (B, 3, T, Hf, Wf), (B, T, 3).

            else:
                output_mask = torch.zeros_like(target_mask)
                output_flags = torch.zeros_like(target_flags)

            # Save some ground truth metadata, e.g. weighted query desirability, to get a feel for
            # this example or dataset.
            # NOTE: diagonal() appends the combined dimension at the END of the shape.
            # https://pytorch.org/docs/stable/generated/torch.diagonal.html
            cur_occl_fracs = occl_fracs[:, query_idx, :, :].diagonal(0, 0, 1)
            cur_occl_fracs = rearrange(cur_occl_fracs, 'T V B -> B T V')  # (B, T, 3).
            cur_desirability = target_desirability[:, query_idx, 0].diagonal(0, 0, 1)  # (B).

            all_occl_fracs.append(cur_occl_fracs)  # (B, T, 3).
            all_desirability.append(cur_desirability)  # (B).
            all_seeker_query_mask.append(seeker_query_mask)  # (B, 1, T, Hf, Wf).
            all_snitch_occl_by_ptr.append(snitch_occl_by_ptr)  # (B, 1, T, Hf, Wf).
            all_full_occl_cont_id.append(full_occl_cont_id)  # (B, T, 2).
            all_target_mask.append(target_mask)  # (B, 3, T, Hf, Wf).
            all_output_mask.append(output_mask)  # (B, 1/3, T, Hf, Wf).
            all_target_flags.append(target_flags)  # (B, T, 3).
            all_output_flags.append(output_flags)  # (B, T, 3) or (1).
            all_aot_loss.append(aot_loss)  # (1, 3) or (1).

        sel_occl_fracs = torch.stack(all_occl_fracs, dim=1)  # (B, Qs, T, 3).
        sel_desirability = torch.stack(all_desirability, dim=1)  # (B, Qs).
        seeker_query_mask = torch.stack(all_seeker_query_mask, dim=1)  # (B, Qs, 1, T, Hf, Wf).
        snitch_occl_by_ptr = torch.stack(all_snitch_occl_by_ptr, dim=1)  # (B, Qs, 1, T, Hf, Wf).
        full_occl_cont_id = torch.stack(all_full_occl_cont_id, dim=1)  # (B, Qs, T, 2).
        target_mask = torch.stack(all_target_mask, dim=1)  # (B, Qs, 3, T, Hf, Wf).
        output_mask = torch.stack(all_output_mask, dim=1)  # (B, Qs, 1/3, T, Hf, Wf).
        target_flags = torch.stack(all_target_flags, dim=1)  # (B, Qs, T, 3).
        output_flags = torch.stack(all_output_flags, dim=1)  # (B, Qs, T, 3) or (1, Qs).
        aot_loss = torch.stack(all_aot_loss, dim=1)  # (1, Qs, T) or (1, Qs).

        # Organize & return relevant info.
        # Ensure that everything is on a CUDA device.
        model_retval = dict()
        model_retval['sel_query_inds'] = sel_query_inds.to(self.device)  # (B, Qs).
        model_retval['sel_occl_fracs'] = sel_occl_fracs.to(self.device)  # (B, Qs, T, 3).
        model_retval['sel_desirability'] = sel_desirability.to(self.device)  # (B, Qs).
        model_retval['seeker_input'] = seeker_input.to(self.device)  # (B, 3, T, Hf, Wf).
        model_retval['seeker_query_mask'] = seeker_query_mask.to(self.device)
        # (B, Qs, 1, T, Hf, Wf).
        model_retval['snitch_occl_by_ptr'] = snitch_occl_by_ptr.to(self.device)
        # (B, Qs, 1, T, Hf, Wf).
        model_retval['full_occl_cont_id'] = full_occl_cont_id.to(self.device)
        # (B, Qs, 1, T, Hf, Wf).
        model_retval['target_mask'] = target_mask.to(self.device)  # (B, Qs, 3, T, Hf, Wf).
        model_retval['output_mask'] = output_mask.to(self.device)  # (B, Qs, 1/3, T, Hf, Wf).
        model_retval['target_flags'] = target_flags.to(self.device)  # (B, Qs, T, 3).
        model_retval['output_flags'] = output_flags.to(self.device)  # (B, Qs, T, 3) or (1, Qs).
        model_retval['aot_loss'] = aot_loss.to(self.device)  # (1, Qs, 3) or (1, Qs).

        return model_retval, cluster_viz

    def forward_ytvos(self, data_retval):
        # NOTE: This is a simplified version of forward_kubric(), so it borders on violating DRY.
        assert self.train_args.which_seeker == 'mask_track_2d'
        assert self.train_args.seeker_input_format == 'rgb'

        within_batch_inds = data_retval['within_batch_idx']
        B = within_batch_inds.shape[0]

        # Retrieve data.
        all_rgb = data_retval['pv_rgb_tf']  # (B, 3, T, Hf, Wf).
        all_segm = data_retval['pv_segm_tf']  # (B, 1, T, Hf, Wf).
        all_rgb = all_rgb.to(self.device)
        all_segm = all_segm.to(self.device)
        inst_count = data_retval['inst_count']  # (B); acts as Qt value per example.
        occl_risk = data_retval['occl_risk']  # (B, M, T, 2) with (percentage, increase).
        inst_area = data_retval['inst_area']  # (B, M, T) in [0, 1].
        query_time = data_retval['query_time']  # (B).

        (T, H, W) = all_rgb.shape[-3:]
        assert T == self.train_args.num_frames
        Qs = 1  # Some videos in this dataset have only a single instance, so ignore --num_queries.

        # Assemble seeker input (which is always a simple copy in this case).
        seeker_input = all_rgb  # (B, 3, T, Hf, Wf).
        qt_idx = query_time[0].item()

        # Sample random queries & construct query & ground truth masks.
        # NOTE: Query handling is intentionally much simpler than kubric; we do not bother with
        # desirability, but we must ensure that the query is actually visible.
        sel_query_inds = torch.zeros(B, dtype=torch.int64, device=self.device)
        seeker_query_mask = torch.zeros_like(all_segm)  # (B, 1, T, Hf, Wf).
        no_gt_frames = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        risky_frames = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        target_mask = torch.zeros_like(seeker_input[:, 0:1])  # (B, 1, T, Hf, Wf).

        for b in range(B):
            query_areas = inst_area[b, :inst_count[b], qt_idx]  # (K) in [0, 1].
            query_choices = torch.nonzero(query_areas >= 0.005).flatten()  # (<= K).
            query_choices = query_choices.detach().tolist()
            assert len(query_choices) > 0, 'No valid queries found for this video.'

            sel_query_inds[b] = np.random.choice(query_choices)
            seeker_query_mask[b, 0, qt_idx] = (all_segm[b, 0, qt_idx] == sel_query_inds[b] + 1)
            target_mask[b, 0] = (all_segm[b, 0] == sel_query_inds[b] + 1)

            # Ignore supervision at frames where no ground truth file exists in the first place.
            no_gt_frames[b] = (all_segm[b, 0] == -1.0).any(dim=-1).any(dim=-1)  # (T).
            target_mask[b, 0, no_gt_frames[b]] = -1.0

            # Ignore supervision at frames with high occlusion risk (this means overriding
            # potential annotations in many cases!).
            occl_thres = 1.0 - self.train_args.ytvos_validity_thres
            risky_frames[b] = (occl_risk[b, sel_query_inds[b], :, 0] > occl_thres)  # (T).
            target_mask[b, 0, risky_frames[b]] -= 3.0
            # Available but unused mask will be -2, background -3, and unavailable -4.

        seeker_query_mask = seeker_query_mask.type(torch.float32).to(self.device)
        target_mask = target_mask.type(torch.float32).to(self.device)

        # Sanity checks.
        assert seeker_query_mask.any(), \
            f'seeker_query_mask all zero? sel_query_inds: {sel_query_inds} qt_idx: {qt_idx}'
        assert (no_gt_frames.int().sum(dim=-1) <= T - 2).all(), \
            f'insufficient available target frames? no_gt_frames: {no_gt_frames}'
        assert target_mask.any(), \
            f'target_mask all zero? sel_query_inds: {sel_query_inds} qt_idx: {qt_idx}'
        assert (target_mask <= 1.0).all(), \
            f'target_mask contains entries > 1? {target_mask.unique().tolist()}'
        assert (target_mask >= -4.0).all(), \
            f'target_mask contains entries < -4? {target_mask.unique().tolist()}'

        # Run seeker to recover hierarchical masks over time.
        if self.train_args.tracker_arch == 'aot':
            # Baseline; no hierarchy, no flags.
            assert 'test' in self.phase, 'We are not training baselines on non-Kubric data for now.'
            output_mask = self.networks['seeker'](seeker_input, seeker_query_mask)
            # (B, 1, T, Hf, Wf).
            output_flags = torch.tensor([0.0], device=self.device)

        else:
            (output_mask, output_flags) = self.networks['seeker'](
                seeker_input, seeker_query_mask)  # (B, 3, T, Hf, Wf), (B, T, 3).

        # Organize & return relevant info.
        # Ensure that everything is on a CUDA device.
        model_retval = dict()
        model_retval['sel_query_inds'] = sel_query_inds.to(self.device)  # (B).
        model_retval['seeker_input'] = seeker_input.to(self.device)  # (B, 3, T, Hf, Wf).
        model_retval['seeker_query_mask'] = seeker_query_mask.to(self.device)  # (B, 1, T, Hf, Wf).
        model_retval['no_gt_frames'] = no_gt_frames.to(self.device)  # (B, T).
        model_retval['risky_frames'] = risky_frames.to(self.device)  # (B, T).
        model_retval['target_mask'] = target_mask.to(self.device)  # (B, 1, T, Hf, Wf).
        model_retval['output_mask'] = output_mask.to(self.device)  # (B, 3, T, Hf, Wf).
        model_retval['output_flags'] = output_flags.to(self.device)  # (B, T, 3).

        return model_retval

    def forward_plugin(self, data_retval):
        # NOTE: This is a simplified version of forward_kubric() and forward_ytvos(), so it borders
        # on violating DRY.
        within_batch_inds = data_retval['within_batch_idx']
        B = within_batch_inds.shape[0]

        assert self.train_args.which_seeker == 'mask_track_2d'
        assert self.train_args.seeker_input_format == 'rgb'

        all_rgb = data_retval['pv_rgb_tf']  # (B, 3, T, Hf, Wf).
        all_query = data_retval['pv_query_tf']  # (B, 1, T, Hf, Wf).
        all_target = data_retval['pv_target_tf']  # (B, 3, T, Hf, Wf).
        all_rgb = all_rgb.to(self.device)
        all_query = all_query.to(self.device)
        all_target = all_target.to(self.device)

        (T, H, W) = all_rgb.shape[-3:]
        assert T == self.train_args.num_frames
        Qs = 1

        # Assemble seeker input (which is always a simple copy in this case).
        seeker_input = all_rgb  # (B, 3, T, Hf, Wf).
        seeker_query_mask = all_query.type(torch.float32).to(self.device)  # (B, 1, T, Hf, Wf).
        target_mask = all_target.type(torch.float32).to(self.device)  # (B, 3, T, Hf, Wf).

        # Sanity checks.
        if not seeker_query_mask.any():
            raise RuntimeError(f'seeker_query_mask all zero?')

        # Run seeker to recover hierarchical masks over time.
        # todo: forward pass -> implement clustering here
        if self.train_args.tracker_arch == 'aot':
            # Baseline; no hierarchy, no flags.
            assert 'test' in self.phase, 'We are not training baselines on non-Kubric data for now.'
            output_mask, cluster_viz = self.networks['seeker'](seeker_input, seeker_query_mask)
            # output_mask = self.networks['seeker'](seeker_input, seeker_query_mask)
            # (B, 1, T, Hf, Wf).
            output_flags = torch.tensor([0.0], device=self.device)

        else:
            (output_mask, output_flags, cluster_viz) = self.networks['seeker'](
                seeker_input, seeker_query_mask)  # (B, 3, T, Hf, Wf), (B, T, 3).

        # Organize & return relevant info.
        # Ensure that everything is on a CUDA device.
        model_retval = dict()
        model_retval['seeker_input'] = seeker_input.to(self.device)  # (B, 3, T, Hf, Wf).
        model_retval['seeker_query_mask'] = seeker_query_mask.to(self.device)  # (B, 1, T, Hf, Wf).
        model_retval['target_mask'] = target_mask.to(self.device)  # (B, 3, T, Hf, Wf).
        model_retval['output_mask'] = output_mask.to(self.device)  # (B, 3, T, Hf, Wf).
        model_retval['output_flags'] = output_flags.to(self.device)  # (B, T, 3).

        return model_retval, cluster_viz

    def process_entire_batch(self, data_retval, model_retval, loss_retval, cur_step, total_step,
                             epoch, progress):
        '''
        Finalizes the training step. Calculates all losses.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        :param cur_step (int): Current data loader index.
        :param total_step (int): Cumulative data loader index, including all previous epochs.
        :param epoch (int): Current epoch index (0-based).
        :param progress (float) in [0, 1]: Total progress within the entire training run.
        :return loss_retval (dict): All loss information.
        '''
        loss_retval = self.losses.entire_batch(
            data_retval, model_retval, loss_retval, cur_step, total_step, epoch, progress)

        return loss_retval
