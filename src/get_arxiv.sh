#!/bin/bash
IDIR=$1
RDIR=$2
SDIR=$3
MAXLINES=$4

if [ ! -d ${IDIR} ]; then
	mkdir ${IDIR}
fi

ARCHIVE=${RDIR}/arxiv.zip
FILENAME=${RDIR}/arxiv.json
OUTFNAME=ARXIV_S
TMP=${RDIR}/temp

#pip install kaggle
# generate authentication token from kaggle
# place that token (json file) in ~/.kaggle/

if ! test -f ${ARCHIVE}; then
	kaggle datasets download -d Cornell-University/arxiv -p ${RDIR}
fi

#Arxiv data has one json file in the archive'
echo "unzipping ${ARCHIVE} (may take minutes)"
unzip -p ${ARCHIVE} > ${FILENAME}
sleep 3

echo "split ${FILENAME} into chunks"
split -l ${MAXLINES} -a 3 -d ${FILENAME} ${IDIR}/${OUTFNAME}

echo "id,authors,title,comments,journal_ref,doi,report_no,categories,abstract,version_num,version_created,last_update" > columns

for SLICE in ${IDIR}/${OUTFNAME}*; do
	echo ${SLICE}
	jq '{id,authors,title,comments,"journal-ref",doi,"report-no",categories,abstract,"version":.versions[-1].version,"created":.versions[-1].created,update_date}' ${SLICE}> ${TMP}
	vim -es -c '%s/\\\+n/ /g' -cwq ${TMP}
	jq -r '[.[]] | @csv' ${TMP} > ${SLICE}
	cat columns ${SLICE} > ${TMP}
	mv ${TMP} ${SLICE}
done | tqdm --total $(ls ${IDIR}/${OUTFNAME}* | wc -l) --unit files --desc 'Parsing JSON' >> /dev/null

rm columns
