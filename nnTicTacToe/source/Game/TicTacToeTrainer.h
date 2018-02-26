#pragma once

#include <memory>

#include "GameLogic.h"
#include "NeuralNetwork/NodeNetwork.h"
#include "NeuralNetwork/ParameterManager.h"
#include "Player.h"

namespace Game
{
    struct ScoreSet
    {
        int invalidCount = 0;
        int wonCount = 0;
        int lostCount = 0;
        int tiedCount = 0;

        std::vector<double> scores;
    };

    class TicTacToeTrainer
    {
    public:
        TicTacToeTrainer();
        ~TicTacToeTrainer();

    public:
        bool setup();
        void run();

        void handleTrainingIteration(bool requiresFurtherEvolution);

    private:
        void describeTrainer() const;
        void describeScoreForId(int id) const;
        void playMatch(BasePlayer& playerA, BasePlayer& playerB);
        GameState playOneTurn(BasePlayer& player, bool firstPlayer);
        double computeMatchScore(BasePlayer& player, int numTurns, GameState finalGameState);
        void addScore(const BasePlayer& player, double score, GameState playerGameState);
        double getAverageScoreForId(int id) const;
        double getOutcomeRatioScoreForId(int id) const;
        void handleParamSetEvolution();

    private:
        bool m_initialized = false;

        double m_minParamValue;
        double m_maxParamValue;
        int m_numIterations = 1;
        int m_numParamSets = 1;
        int m_numMatches = 1;

        std::shared_ptr<TicTacToeLogic> m_gameLogic;
        std::shared_ptr<NeuralNetwork::ParameterManager> m_paramManager;
        std::shared_ptr<NeuralNetwork::NodeNetwork> m_nodeNetwork;
        std::map<int, ScoreSet> m_scoreMap;
    };
}
