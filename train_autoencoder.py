import argparse
from solver import SolverAutoencoder

def main(args):
    solver = SolverAutoencoder(args)
    solver.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "smnist_exp", type=str)
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--data_root", default="data", type=str)
    parser.add_argument("--epoch_size", default=600, type=int)
    parser.add_argument("--niters", default=200, type=int)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--checkpoint_freq", default=3, type = int)
    parser.add_argument("--summary_freq", default=1, type = int)
    parser.add_argument("--max_checkpoints", default=5, type=int)
    parser.add_argument("--lr", default=0.002, type=float)
    parser.add_argument("--beta1", default=0.5, type=float)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--z_dims", default=5, type=int)
    parser.add_argument("--g_dims", default=128, type=int)
    parser.add_argument("--input_frames", default=5, type=int)
    parser.add_argument("--target_frames", default=10, type=int)
    parser.add_argument("--sim_loss", default="emily", type=str)
    parser.add_argument("--sd_loss", default="js", type=str, choices = ["js","emily","smile"])
    parser.add_argument("--sim_weight", default=1.0, type=float)
    parser.add_argument("--sd_weight", default=0.0001, type=float)
    parser.add_argument("--num_channels", default=3, type=int)
    parser.add_argument("--color", dest="color", action="store_true")
    parser.add_argument("--no_color", dest="color", action="store_false")
    parser.set_defaults(color=True)
    parser.add_argument("--skips", dest = "skips", action = "store_true")
    parser.add_argument("--no_skips", dest = "skips", action="store_false")
    parser.set_defaults(skips=False)
    parser.add_argument("--dataset", default = "mnist", type = str)
    parser.add_argument("--content_lstm", dest = "content_lstm", action = "store_true")
    parser.add_argument("--no_content_lstm", dest = "content_lstm", action = "store_false")
    parser.set_defaults(content_lstm=False)
    parser.add_argument("--recon_loss_type", default = "mse", type = str)
    parser.add_argument("--rotate_sprites", dest = "rotate_sprites", action="store_true")
    parser.add_argument("--no_rotate_sprites", dest = "rotate_sprites", action="store_false")
    parser.set_defaults(rotate_sprites=True) #used only for dsprites dataset
    parser.add_argument("--normalize_position", dest = "normalize_position", action="store_true")
    parser.add_argument("--no_normalize_position", dest = "normalize_position", action="store_false")
    parser.set_defaults(normalize_position=False) #used only for dsprites dataset

    args = parser.parse_args()
    main(args)

