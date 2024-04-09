
auto_mass_factor = np.clip(self.random_state.normal(0.0, 0.2), 0.5, 1.5)

# DEBUG / TEMP:
# exp16d:
# obj.mass = np.sqrt(gpt_mass * auto_mass) * (1.0 + extra_noise)
# exp16e:
# obj.mass = np.cbrt(gpt_mass * gpt_mass * auto_mass)
# exp16f:
# obj.mass = auto_mass
# exp16g:
# obj.mass = gpt_mass
# ^ ALL ABOVE ARE BUGGED because was accidentally using v3_mass instead of v4_mass!

# exp16h:
# obj.mass = gpt_mass
# exp16i:
# obj.mass = auto_mass
