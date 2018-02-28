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
        double finalScore = 0;
    };

    class TicTacToeTrainer
    {
    public:
        TicTacToeTrainer() = default;
        ~TicTacToeTrainer();

    public:
        bool setup();
        void run();

        void handleTrainingIteration(int iteration);

    private:
        bool readConfigValues();
        void describeTrainer() const;
        void describeScoreForId(int id) const;
        void playMatch(BasePlayer& playerA, BasePlayer& playerB);
        GameState playOneTurn(BasePlayer& player, bool firstPlayer);
        double computeMatchScore(BasePlayer& player, int numTurns, GameState finalGameState);
        void addScore(const BasePlayer& player, double score, GameState playerGameState);
        double computeFinalScore(int id);
        double getAverageScoreForId(int id) const;
        double getOutcomeRatioScoreForId(int id) const;
        bool getScoreSetForId(int id, ScoreSet& scoreSet) const;
        void handleParamSetEvolution();
        void dumpTrainingStats() const;
        void dumpBestSetImprovementStats() const;

    private:
        bool m_initialized = false;

        ParameterManagerData m_paramData; /// collection of data needed by the parameter manager
        int m_numIterations = 1; /// number of training iterations
        int m_numParamSets = 5; /// number of concurrently tried parameter sets

        /// number of matches run to compute a score for each parameter set
        /// actually, we run twice this amount (trying both as first and second player)
        int m_numMatches = 10; 

        std::vector<int> m_numHiddenNodes; /// number of nodes within each hidden layer

        std::shared_ptr<TicTacToeLogic> m_gameLogic;
        std::shared_ptr<NeuralNetwork::ParameterManager> m_paramManager;
        std::shared_ptr<NeuralNetwork::NodeNetwork> m_nodeNetwork;
        std::map<int, ScoreSet> m_scoreMap;
        std::map<int, std::vector<int>> m_idsPerIteration;
        std::vector<std::vector<CellState>> m_gameStateCollection;
    };
}
