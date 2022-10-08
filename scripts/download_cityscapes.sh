#! /bin/bash
HELP=false
USERNAME=""
PASSWORD=""

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -u|--username)
            shift
            USERNAME="$1"
            ;;
        -p|--password)
            shift
            PASSWORD="$1"
            ;;
        *)
            HELP=true
            ;;
    esac
    shift
done

if [[ $HELP = true ]] || [[ $USERNAME = "" ]] || [[ $PASSWORD = "" ]]
then
    echo """Usage: ./download_cityscapes.sh -u USERNAME -p PASSWORD

Available options:
------------------
-u --username   Username for the Cityscapes website
-p --password   Password for the Cityscapes website
"""
else
    mkdir -p "/sinergia/cityscapes"

    cd /sinergia/cityscapes

    # login
    wget --keep-session-cookies --save-cookies=cookies.txt --post-data "username=$USERNAME&password=$PASSWORD" https://www.cityscapes-dataset.com/login/

    # get all data
    for i in 1 2 3 4 8 9 10 11 12 28
    do
        wget --load-cookies cookies.txt --content-disposition "https://www.cityscapes-dataset.com/file-handling/?packageID=$i"
    done

    # remove unused files
    rm cookies.txt
    rm index.html

    # unzip data
    for i in $(ls)
    do
        unzip $i
    done
fi
