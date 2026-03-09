from dm_control import composer
from robopianist import music
# 引入带机械手的任务类
from robopianist.suite.tasks import piano_with_shadow_hands 
from robopianist.viewer import launch

def main():
    midi = music.load("TwinkleTwinkleRousseau")
    
    # 手动实例化带手的任务
    task = piano_with_shadow_hands.PianoWithShadowHands(
        midi=midi,
        change_color_on_activation=True,
        control_timestep=0.05,
    )

    env = composer.Environment(task=task, strip_singleton_obs_buffer_dim=True)
    launch(env)

if __name__ == "__main__":
    main()
