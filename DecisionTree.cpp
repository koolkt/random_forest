#include "DecisionTree.h"

#include <algorithm>
#include <cmath>

namespace dt {

using std::sort;

float evalTree(DecisionTree* tree, const FeatureVector& features) {
  // assert(tree);
  if (!tree->left) {
    return tree->value;
  }
  if (features[tree->splitFeature] < tree->splitValue) {
    return evalTree(tree->left.get(), features);
  }
  return evalTree(tree->right.get(), features);
}

float entropy(size_t p, size_t n) {
  // Avoid taking logarithm of zero when either p or n are zero
  ++p;
  ++n;
  return - p * log((float) p / (p + n)) - n * log((float) n / (p + n));
}

float infoGain(size_t p, size_t n, size_t p1, size_t n1) {
  return entropy(p, n) - entropy(p1, n1) - entropy(p - p1, n - n1);
}

unique_ptr<DecisionTree> trainTree(ExampleIt first,
                                   ExampleIt last,
                                   size_t maxDepth,
                                   float minGain) {
  if (first >= last) return nullptr;
  if (maxDepth == 0) return nullptr;

  size_t numFeatures = first->features.size();
  size_t p = 0, n = 0;
  for (auto it = first; it != last; ++it) {
    if (it->label) {
      ++p;
    } else {
      ++n;
    }
  }

  float bestGain = -1;
  size_t splitIndex;
  size_t splitFeature;
  float splitValue;

  for (size_t i = 0; i < numFeatures; ++i) {
    // Sort by feature i
    sort(first, last, [=](const Example& a, const Example& b) {
        return a.features[i] < b.features[i];
      });

    size_t p1 = 0, n1 = 0;
    for (ExampleIt it = first; it != last; ++it) {
      if (it != first && it->features[i] != (it - 1)->features[i]) {
        float gain = infoGain(p, n, p1, n1);
        if (gain > bestGain) {
          bestGain = gain;
          splitIndex = it - first;
          splitFeature = i;
          splitValue = it->features[i];
        }
      }
      if (it->label) {
        ++p1;
      } else {
        ++n1;
      }
    }
  }

  unique_ptr<DecisionTree> tree(new DecisionTree);
  tree->value = (float) p / (p + n);
  
  if (bestGain > minGain) {
    tree->splitFeature = splitFeature;
    tree->splitValue = splitValue;
    
    sort(first, last, [=](const Example& a, const Example& b) {
        return a.features[splitFeature] < b.features[splitFeature];
      });

    tree->left = std::move(trainTree(first,
                                     first + splitIndex,
                                     maxDepth - 1,
                                     minGain));
    tree->right = std::move(trainTree(first + splitIndex,
                                      last,
                                      maxDepth - 1,
                                      minGain));
  }

  return tree;
}

float validateTree(DecisionTree* tree, ExampleIt first, ExampleIt last) {
  size_t correct = 0;
  for (auto it = first; it != last; ++it) {
    float value = evalTree(tree, it->features);
    if ((value >= 0.5) == it->label) {
      ++correct;
    }
  }
  return (float) correct / (last - first);
}

}
