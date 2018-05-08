import org.aksw.palmetto.data.SegmentationDefinition;
import org.aksw.palmetto.subsets.Segmentator;
import com.carrotsearch.hppc.BitSet;

public class PairwiseTopicComparingSegmentator implements Segmentator {
  @Override
  public SegmentationDefinition getSubsetDefinition(int wordset_size) {
    if ((wordset_size % 2)!=0)
      throw new IllegalArgumentException("word set size is not even!");
    int single_topic_size = wordset_size / 2;
    int secong_topic_lowest_bit = 1<<single_topic_size;
    int conditions[][] = new int[wordset_size][single_topic_size];
    int segments[] = new int[wordset_size];
    int cond_bit, cond_pos, bit=1, pos=0;
    int mask = (1<<wordset_size)-1;
    BitSet counts = new BitSet(1<<wordset_size);
    while (bit < mask) {
      segments[pos]=bit;
      counts.set(bit);
      cond_pos = 0;
      if (pos < single_topic_size) {
        cond_bit = secong_topic_lowest_bit;
        while (cond_bit < mask) {
          counts.set(bit + cond_bit);
          conditions[pos][cond_pos] = cond_bit;
          ++cond_pos;
          cond_bit = cond_bit << 1;
        }
      } else {
        cond_bit = 1;
        while (cond_bit<secong_topic_lowest_bit) {
          counts.set(bit+cond_bit);
          conditions[pos][cond_pos] = cond_bit;
          ++cond_pos;
          cond_bit = cond_bit << 1;
        }
      }
      bit = bit << 1;
      ++pos;
    }
    return new SegmentationDefinition(segments, conditions, counts);
  }

  @Override
  public String getName() {
    return "one-topic";
  }
}
