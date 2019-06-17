import os
import argparse

def main():
    # default values:
    syn_mode = 'p1'
    num_epoch = 200
    num_channels = 3
    num_videos = 233
    verbose = 0
    number = 100

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="synthetic mode: i-interp; p-predict; m-motion deblur;"
                                             " + unknown frame #")
    parser.add_argument("-e", "--num_epochs", help="number of epochs", type=int)
    parser.add_argument("--save_dir", help="save directory")
    parser.add_argument("--info_file", help="path to adobe-nfs-split.txt file")
    parser.add_argument("-c", "--color", help="number of color channels", type=int)
    parser.add_argument("--clips", help="number of clips", type=int)
    parser.add_argument("-n", "--number", help="number of lines", type=int)
    args = parser.parse_args()
    if args.mode:
        syn_mode = args.mode
    if args.num_epochs:
        num_epoch = args.num_epochs
    if args.save_dir:
        save_dir = args.save_dir
    if args.info_file:
        info_file = args.info_file
    if args.color:
        num_channels = args.color
    if args.clips:
        num_videos = args.clips
    if args.number:
        number = args.number

    input_string = '-m %s -e %s -c %s --clips %s -v 1' % (syn_mode, num_epoch, num_channels, num_videos)
    if os.path.exists('Batch-%s.bat' % input_string):
        os.remove('Batch-%s.bat' % input_string)
    with open('Batch-%s.bat' % input_string, 'a') as f:
        f.write('cd F:\Winston\e-vfs\int-event-fusion\sample_preparation\n')
        f.write('CALL conda.bat activate tensorflow-gpu\n')
        for i in range(args.number):
            f.write('python prepare_training_samples.py %s\n' % input_string)
        f.write('pause')

if __name__ == '__main__':
    main()