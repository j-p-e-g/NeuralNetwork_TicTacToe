#pragma once

#include <map>
#include <vector>

namespace NeuralNetwork
{
    struct ParamSet
    {
        double score = 0.0;
        std::vector<double> params;
    };

    class ParameterManager
    {
    public:
        ParameterManager() = delete;
        ParameterManager(int num, double minValue, double maxValue);

    public:
        bool readDataFromFile();
        bool dumpDataToFile() const;
        void fillWithRandomValues(ParamSet& pset) const;
        void addNewParamSet(const ParamSet& pset);

    private:
        std::map<int, ParamSet> m_paramSets;
        int m_highestId;

        int m_paramAmount;
        double m_minValue;
        double m_maxValue;
    };
}
