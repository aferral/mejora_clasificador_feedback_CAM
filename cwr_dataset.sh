


if hash unzip 2>/dev/null; then
    wget "https://www.dropbox.com/s/domvk3dnt337idx/CW96Scalograms.zip?dl=1" -O "./temp/CW96Scalograms.zip"
    unzip "./temp/CW96Scalograms.zip" -d "./temp/"
    rm "./temp/CW96Scalograms.zip"

    python -m tf_records_parser.cwr_parser ./temp/CW96Scalograms/ ./temp/CWR_records

else
    echo >&2 "I require unzip (install with sudo apt-get install unzip) but it's not installed.  Aborting.";
    exit 1;
fi




