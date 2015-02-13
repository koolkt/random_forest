// Dataset from http://archive.ics.uci.edu/ml/datasets/Hepatitis

#include <algorithm>
#include <fstream>
#include <iostream>

#include "../DecisionTree.h"
#include "../RandomForest.h"

using namespace std;

using namespace dt;

const size_t NUM_FEATURES = 19;

int main() {
  vector<Example> examples;

  ifstream input("hepatitis_preproc.data");

  bool label;
  while (input >> label) {
    Example ex;
    ex.label = label;
    ex.features.resize(NUM_FEATURES);
    for (size_t i = 0; i < NUM_FEATURES; ++i) {
      input >> ex.features[i];
    }
    examples.push_back(ex);
  }
  
  size_t numTraining = 0.7 * examples.size();

  srand(time(0));
  random_shuffle(begin(examples), end(examples));
  vector<Example> trainingSet(begin(examples), begin(examples) + numTraining);
  vector<Example> testSet(begin(examples) + numTraining, end(examples));

  cout << "DECISION TREES: " << endl;
  
  for (size_t maxDepth = 1; maxDepth <= 10; ++maxDepth) {
    auto tree = trainTree(begin(trainingSet), end(trainingSet), maxDepth);
    float trainingAccuracy = validateTree(tree.get(),
                                          begin(trainingSet),
                                          end(trainingSet));
    float testAccuracy = validateTree(tree.get(),
                                      begin(testSet),
                                      end(testSet));
    cout << "Max depth = " << maxDepth
         << " => training accuracy = " << trainingAccuracy
         << ", test accuracy = " << testAccuracy << endl;
  }

  cout << "RANDOM FOREST:" << endl;
  
  auto forest = trainForest(begin(trainingSet), end(trainingSet));
  float trainingAccuracy = validateForest(forest,
                                          begin(trainingSet),
                                          end(trainingSet));
  float testAccuracy = validateForest(forest,
                                      begin(testSet),
                                      end(testSet));
  cout << "Training accuracy = " << trainingAccuracy
       << ", test accuracy = " << testAccuracy << endl;
}
