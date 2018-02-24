#include "stdafx.h"
#include <memory>
#include <iostream>

#include "NeuralNetwork/NodeNetwork.h"
#include "NeuralNetwork/ParameterManager.h"

using namespace NeuralNetwork;

void buildAndDestroyNetwork()
{
    std::shared_ptr<NodeNetwork> network = std::make_shared<NodeNetwork>();
    if (!network->createNetwork(3, 1, std::vector<int>()))
    {
        std::cerr << "Network creation failed!" << std::endl;
    }

    network->assignInputValues(std::vector<double>({ 1, -0.5, -1 }));

    std::queue<double> values = std::queue<double>({ 0.2, 0.7, -0.3, 1.2 });
    if (network->assignParameters(values))
    {
        network->computeValues();

        std::vector<double> outputValues;
        network->getOutputValues(outputValues);
        std::cout << "Output values: ";
        for (const auto& val : outputValues)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    else
    {
        std::cerr << "Assigning parameters failed!" << std::endl;
    }

    network->destroyNetwork();
}

void setupParameterManager()
{
    ParameterManager pm(5, -10.0, 10.0);
    pm.readDataFromFile();
    pm.dumpDataToFile();

    ParamSet pset;
    pm.fillWithRandomValues(pset);
    pm.addNewParamSet(pset);
}

int main()
{
    //buildAndDestroyNetwork();
    setupParameterManager();
    return 0;
}
