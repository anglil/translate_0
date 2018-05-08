import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.stream.Stream;

public class jk {
  public static void main(String[] args) {
    try (Stream<String> lines = Files.lines(Paths.get("myres.tmp"))) {
      int pos_in_file = 2;
      String line = lines.skip(pos_in_file).findFirst().get();
      System.out.println(line);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
