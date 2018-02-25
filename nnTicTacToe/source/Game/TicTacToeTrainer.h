#pragma once

#include <memory>

#include "GameLogic.h"
#include "NeuralNetwork/NodeNetwork.h"
#include "NeuralNetwork/ParameterManager.h"

namespace Game
{
    class TicTacToeTrainer
    {
    public:
        TicTacToeTrainer();
        ~TicTacToeTrainer();

    public:
        bool setup();
        void run();

    private:
        double playMatch();
        double computeMatchScore(int numTurns, GameState finalGameState);
        GameState playOneTurn();

    private:
        bool m_initialized = false;

        double m_minParamValue;
        double m_maxParamValue;
        int m_numParamSets = 1;
        int m_numMatches = 1;

        std::shared_ptr<TicTacToeLogic> m_gameLogic;
        std::shared_ptr<NeuralNetwork::ParameterManager> m_paramManager;
        std::shared_ptr<NeuralNetwork::NodeNetwork> m_nodeNetwork;
    };
}
