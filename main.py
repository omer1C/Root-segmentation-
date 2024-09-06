# import Arbidiopsis
# import Bell_Pepper
import argparse
import os

import Arbidiopsis
import Bell_Pepper


def main(args):
    """Main run the code and demonstrate."""
    if args.plant_type == "Bell_Pepper":
        base_path = r''
        mid_path = os.path.join(base_path, args.path)
        full_path = os.path.join(mid_path,'FinalProject/Bell_Images/')
        Bell_Pepper.Bell_Pepper_active(full_path)
    elif args.plant_type == "Arbidiopsis":
        base_path = r''
        mid_path = os.path.join(base_path, args.path)
        full_path = os.path.join(mid_path,'FinalProject/Arb_Images/')
        Arbidiopsis.Arbidiopsis_active(full_path)
    else:
        print('Sorry plant type is not correct, please try Bell_Pepper or Arbidiopsis.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing soil and plant quality through segmentation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General settings.
    parser.add_argument("--plant_type", type=str, default="Bell_Pepper",
                        help="Please enter Bell_Pepper or Arbidiopsis to activate the code.")

    parser.add_argument("--path", type=str,
                        help="Path to the project on your computer, should look like : "
                             "/Users/YOUR_USER_NAME/PycharmProjects.")

    args = parser.parse_args()
    main(args)