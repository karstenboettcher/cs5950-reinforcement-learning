from absl import app
from dopamine.utils import example_viz_lib


def main(_):
    example_viz_lib.run(agent='rainbow',
                        game='Pong',
                        num_steps=2000,
                        root_dir='D:\\Homework\\CS 5950\\Pong_Rainbow',
                        restore_ckpt='D:\\Homework\\CS 5950\\Pong_Rainbow\\checkpoints\\tf_ckpt-101',
                        use_legacy_checkpoint=False)


if __name__ == '__main__':
    app.run(main)