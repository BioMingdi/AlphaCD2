python AlphaCD2_predict.py \
  --input-txt test.txt \
  --output-tsv test_result.tsv \
  --ontarget-script AlphaCD2_ontarget.py \
  --manifest bilstm_manifest.json \
  --esm-checkpoint $path/esmc_600m_2024_12_v0.pth \
  --specificity-checkpoint auxiliary_metrics.pt
