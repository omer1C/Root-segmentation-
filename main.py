import argparse
import os
import Arbidiopsis
import Bell_Pepper


def main(args):
    """Main run the code and demonstrate."""
    if args.plant_type == "bell_pepper":
        mid_path = os.getcwd()
        full_path = os.path.join(mid_path, 'bell_images/')
        print(full_path)
        Bell_Pepper.Bell_Pepper_active(full_path, args.save_path)
    elif args.plant_type == "arbidiopsis":

        mid_path = os.getcwd()
        full_path = os.path.join(mid_path, 'arb_images/')
        Arbidiopsis.Arbidiopsis_active(full_path, args.save_path)
    else:
        print('Sorry plant type is not correct, please try bell_pepper or arbidiopsis.')

if __name__ == "__main__":
    default_save_path = os.getcwd()
    default_save_path = os.path.join(default_save_path, 'example')
    parser = argparse.ArgumentParser(description="Analyzing soil and plant quality through segmentation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General settings.
    parser.add_argument("--plant_type", type=str, default="bell_pepper",
                        help="Please enter bell_pepper or arbidiopsis to activate the code.")
    parser.add_argument("--save_path", type=str, default=default_save_path,
                        help="Please enter the location you want to save the results.")
    args = parser.parse_args()
    main(args)