import os
import boto3
import argparse

# Define the series to download
SERIES_MAP = {
   'Diffusion': 'T1w/Diffusion/',
   'T1wRestoreBrain': 'T1w/T1w_acpc_dc_restore_brain.nii.gz',
   'l_PialSurf': 'T1w/{subject}/surf/lh.pial',
   'l_WhiteSurf': 'T1w/{subject}/surf/lh.white',
   'r_PialSurf': 'T1w/{subject}/surf/rh.pial',
   'r_WhiteSurf': 'T1w/{subject}/surf/rh.white',
   'RibbonMGZ': 'T1w/{subject}/mri/ribbon.mgz',
   'AsegMGZ': 'T1w/{subject}/mri/aseg.mgz',
    'wm':'T1w/{subject}/mri/wm.mgz',
    'l_reg':'T1w/{subject}/surf/lh.sphere.reg',
    'r_reg':'T1w/{subject}/surf/rh.sphere.reg',
}

# Main collect and download function
def collect_and_download(out_dir, subject, series=SERIES_MAP.keys()):
    # Initialize S3 access
    s3_bucket_name = 'hcp-openaccess'
    s3_prefix = 'HCP_1200'
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(s3_bucket_name)

    for serie in series:
        series_path = SERIES_MAP[serie].format(subject=subject)
        if series_path.endswith('/'):  # Handle directories
            prefix = f"{s3_prefix}/{subject}/{series_path}"
            for obj in bucket.objects.filter(Prefix=prefix):
                download_file = os.path.join(out_dir, obj.key.replace(s3_prefix, '').lstrip('/'))
                download_dir = os.path.dirname(download_file)
                if not os.path.exists(download_dir):
                    os.makedirs(download_dir)
                if not os.path.exists(download_file):
                    print(f'Downloading to: {download_file}')
                    bucket.download_file(obj.key, download_file)
        else:  # Handle individual files
            s3_key = f"{s3_prefix}/{subject}/{series_path}"
            download_file = os.path.join(out_dir, s3_key.replace(s3_prefix, '').lstrip('/'))
            download_dir = os.path.dirname(download_file)
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            if not os.path.exists(download_file):
                print(f'Downloading to: {download_file}')
                bucket.download_file(s3_key, download_file)
                
# Read subject IDs from file
def read_subject_ids(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

# Make module executable
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--out_dir', required=True, type=str, help='Path to local folder to download files to')
    parser.add_argument('-f', '--file_path', required=True, type=str, help='Path to file containing subject IDs')

    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    file_path = os.path.abspath(args.file_path)
    subject_ids = read_subject_ids(file_path)

    # Download data for each subject
    for subject in subject_ids:
        print(f'Downloading data for subject: {subject}')
        collect_and_download(out_dir=out_dir, subject=subject)
