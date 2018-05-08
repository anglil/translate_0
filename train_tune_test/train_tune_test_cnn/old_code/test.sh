for lang in vie; do
  for dim in 128 256 512; do
    for layer in 2; do
      for kernel in 2 3 4; do
        for bpe in 8000 32000 inf 8000-8000; do
          sh run.sh $lang eng 2 $dim $layer $kernel $bpe test
        done
      done
    done
  done
done
