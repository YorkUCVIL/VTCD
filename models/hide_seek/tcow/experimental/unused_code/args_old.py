

    parser.add_argument('--try_queries', default=1024, type=int,
                        help='For query-based trackers, number of points to track per example in '
                        'the data loader.')


    parser.add_argument('--samples_per_pixel', default=16, type=int,
                        help='Kubric / Blender rendering quality.')
    parser.add_argument('--avg_static_objects', default=8, type=int,
                        help='Number of mostly static objects in every Kubric scene / video clip.')
    parser.add_argument('--avg_dynamic_objects', default=4, type=int,
                        help='Number of mostly dynamic objects in every Kubric scene / video clip.')

    parser.add_argument('--ytvos_border_neutral', default=16, type=int,
                        help='Padding (in pixels) around positive target mask borders where to '
                        'apply no supervision, since occlusions may cause them to be ambiguous '
                        'areas of the video.')

    # Hider (simulator) options.
    parser.add_argument('--which_hider', default='none', type=str,
                        help='How the hider operates (snitch / none / ...).')
    parser.add_argument('--policy_arch', default='resnet50', type=str,
                        help='Neural network architecture for hider (timesformer / resnet50).')
    parser.add_argument('--value_arch', default='resnet18', type=str,
                        help='Neural network architecture for hider value function (resnet18).')
    parser.add_argument('--hider_frames', default=2, type=int,
                        help='How many input frames the hider perceives before taking action.')
    parser.add_argument('--hider_output', default='init_pos', type=str,
                        help='What hider predicts (init_pos / init_vel / etc).')
    parser.add_argument('--action_space', default=16, type=int,
                        help='Size of the policy output vector, i.e. how many different actions '
                        'the hider can take.')


    parser.add_argument('--bootstrap_epochs', default=999, type=int,
                        help='How long to initially use the pregenerated dataset for.')
    parser.add_argument('--seeker_steps', default=10, type=int,
                        help='How many consecutive episodes to train just the seeker for.')
    parser.add_argument('--hider_steps', default=0, type=int,
                        help='How many consecutive episodes to train just the hider for.')



    parser.add_argument('--gauss_reg_lw', default=3e-5, type=float,
                        help='For point_track only. L2 regularization loss weight to mitigate '
                        'exploding Gaussian NLL loss.')
    parser.add_argument('--future_weight', default=3.0, type=int,
                        help='Frames corresponding to forecasted trajectories (of objects or '
                        'points) are considered this factor times more important for the loss.')
    parser.add_argument('--out_of_frame_weight', default=1.0, type=float,
                        help='Frames corresponding to out-of-frame trajectories (of objects or '
                        'points) are considered this factor times more important for the loss. '
                        'NOTE: The sum weight is applied if both future and out of frame '
                        'occur, but this factor always fully replaces occluded when that occurs.')