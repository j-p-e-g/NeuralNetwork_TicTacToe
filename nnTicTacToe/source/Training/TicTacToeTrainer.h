#pragma once

#include <memory>

#include "Game/GameLogic.h"
#include "Game/Player.h"
#include "NeuralNetwork/NodeNetwork.h"
#include "NeuralNetwork/ParameterManager.h"
#include "Training/BaseTrainer.h"

namespace Training
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
        : public BaseTrainer
    {
    public:
        TicTacToeTrainer(std::shared_ptr<Game::GameLogic>& gameLogic);
        ~TicTacToeTrainer();

    public:
        bool setupTrainingData() override;
        bool setupTrainingMethod() override;
        NetworkSizeData getNetworkSizeData() const override;
        void run() override;
        void handleNetworkComputation(int id, bool isLastIteration) override;

    protected:
        std::string getName() const override { return "TicTacToeTrainer"; }

    private:
        // scoring
        void describeScoreForId(int id) const override;
        double computeFinalScore(int id) override;

        void playMatch(Game::BasePlayer& playerA, Game::BasePlayer& playerB);
        GameState playOneTurn(Game::BasePlayer& player, bool firstPlayer);
        double computeMatchScore(Game::BasePlayer& player, int numTurns, GameState finalGameState);
        void addScore(const Game::BasePlayer& player, double score, GameState playerGameState);
        double getAverageScoreForId(int id) const;
        double getOutcomeRatioScoreForId(int id) const;
        bool getScoreSetForId(int id, ScoreSet& scoreSet) const;
        void dumpTrainingStats() const;
        void dumpBestSetImprovementStats() const;

    private:
        std::shared_ptr<Game::GameLogic> m_gameLogic;
        std::map<int, ScoreSet> m_scoreMap;
        std::vector<std::vector<CellState>> m_gameStateCollection;
    };
}
