# import Arbidiopsis
# import Bell_Pepper
import argparse

import Arbidiopsis
import Bell_Pepper


def main(args):
    """Main run the code and demonstrate."""
    if args.plant_type == "Bell_Pepper":
        Bell_Pepper.Bell_Pepper_active()
    elif args.plant_type == "Arbidiopsis":
        Arbidiopsis.Arbidiopsis_active()
    else:
        print('Sorry plant type is not correct, please try Bell_Pepper or Arbidiopsis.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing soil and plant quality through segmentation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General settings.
    parser.add_argument("--plant_type", type=str, default="Bell_Pepper",
                        help="Please enter Bell_Pepper or Arbidiopsis to activate the code.")

    args = parser.parse_args()
    main(args)