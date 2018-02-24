#include "stdafx.h"
#include <memory>
#include <iostream>

#include "Game/GameLogic.h"
#include "NeuralNetwork/NodeNetwork.h"
#include "NeuralNetwork/ParameterManager.h"

using namespace Game;
using namespace NeuralNetwork;

void buildAndDestroyNetwork()
{
    NetworkSizeData sizeData;
    sizeData.numInputNodes = 3;
    sizeData.numOutputNodes = 1;
    sizeData.numHiddenNodes = std::vector<int>();

    std::shared_ptr<NodeNetwork> network = std::make_shared<NodeNetwork>();
    if (!network->createNetwork(sizeData))
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
    ParameterManagerData pmData;
    pmData.numParams = 5;
    pmData.minValue = -1.0;
    pmData.maxValue = 1.0;

    ParameterManager pm(pmData);
    pm.readDataFromFile();
    pm.dumpDataToFile();

    ParamSet pset;
    pm.fillWithRandomValues(pset);
    pm.addNewParamSet(pset);
}

GameState playOneTurn(TicTacToeLogic& tttLogic, std::shared_ptr<NodeNetwork>& network)
{
    std::vector<double> inputValues;
    tttLogic.getNodeNetworkInputValues(inputValues);
    network->assignInputValues(inputValues);

    network->computeValues();

    std::vector<double> outputValues;
    int bestResult = network->getOutputValues(outputValues);

    std::cout << "Output: ";
    for (auto val : outputValues)
    {
        std::cout << val << "  ";
    }

    std::cout << "--> best index: " << bestResult << std::endl;

    if (!tttLogic.isValidMove(0, bestResult))
    {
        std::cerr << "Invalid move" << std::endl;
        return GS_GAMEOVER_LOST;
    }

    tttLogic.applyMove(0, bestResult);

    const GameState state = tttLogic.evaluateBoard();
    std::cout << "Outcome: " << GameLogic::getGameStateDescription(state).c_str() << std::endl;

    return state;
}

void setupTicTacToeTest()
{
    TicTacToeLogic tttLogic;

    NetworkSizeData sizeData;
    tttLogic.getRequiredNetworkSize(sizeData);

    std::shared_ptr<NodeNetwork> network = std::make_shared<NodeNetwork>();
    if (!network->createNetwork(sizeData))
    {
        std::cerr << "Network creation failed!" << std::endl;
    }

    ParameterManagerData pmData;
    pmData.numParams = network->getNumParameters();
    pmData.minValue = -10.0;
    pmData.maxValue = 10.0;

    ParameterManager pm(pmData);
    ParamSet pset;
    pm.fillWithRandomValues(pset);

    std::queue<double> params;
    for (const auto& val : pset.params)
    {
        params.push(val);
    }

    network->assignParameters(params);

    int turnCount = 1;
    GameState state = GS_ONGOING;
    do
    {
        std::cout << "Turn " << turnCount << ": " << std::endl;
        state = playOneTurn(tttLogic, network);
        turnCount++;
    }
    while (state == GS_ONGOING);

    network->destroyNetwork();
}

int main()
{
    //buildAndDestroyNetwork();
    //setupParameterManager();
    setupTicTacToeTest();
    return 0;
}
