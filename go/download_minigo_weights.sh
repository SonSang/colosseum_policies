# Name of bucket to download weight files from
BUCKET_NAME=minigo-pub

# Version and name of model : Change these to download different weight files
# Currently, version 17, 16, 15, 12, 10, 9
MODEL_VERSION=v17-19x19
MODEL_NAME=001003-leviathan

# Create download folder
HERE=$( cd "$(dirname "$0")" ; pwd )
DOWNLOAD_DIR=${HERE}/minigo-models

if [ ! -d ${DOWNLOAD_DIR} ]; then
    mkdir -p ${DOWNLOAD_DIR}
fi

# Download weight files
gcloud auth application-default login   # This can be omitted once you login to gcloud.
gsutil ls gs://${BUCKET_NAME}/${MODEL_VERSION}/models/${MODEL_NAME}.* | gsutil cp -I ${DOWNLOAD_DIR}