#include "RandomForest.h"

#include <algorithm>
#include <cstdlib>

namespace dt {

float evalForest(const RandomForest& forest, const FeatureVector& features) {
  float total = 0;
  for (auto& tree : forest) {
    total += evalTree(tree.get(), features);
  }
  return total / forest.size();
}

RandomForest trainForest(ExampleIt first,
                         ExampleIt last,
                         size_t numTrees,
                         float sampleRate,
                         size_t maxDepth,
                         float minGain) {
  RandomForest forest;
  size_t numExamples = sampleRate * (last - first);
  srand(time(0));
  for (size_t i = 0; i < numTrees; ++i) {
    std::random_shuffle(first, last);
    forest.emplace_back(std::move(trainTree(first,
                                            first + numExamples,
                                            maxDepth,
                                            minGain)));
  }
  return forest;
}

float validateForest(const RandomForest& forest,
                     ExampleIt first,
                     ExampleIt last) {
  size_t correct = 0;
  for (auto it = first; it != last; ++it) {
    float value = evalForest(forest, it->features);
    if ((value >= 0.5) == it->label) {
      ++correct;
    }
  }
  return (float) correct / (last - first);
}

}
