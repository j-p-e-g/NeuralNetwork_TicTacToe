#include "stdafx.h"
#include "CppUnitTest.h"

#include "Game/TicTacToeTrainer.h"
#include "NeuralNetwork/ParameterManager.h"

#include <set>

namespace TicTacToeTrainerTest
{
    using namespace Microsoft::VisualStudio::CppUnitTestFramework;
    using namespace Game;

    TEST_CLASS(TicTacToeTrainer_Test)
    {
    public:
        // -----------------------------------------
        // TicTacToeTrainer
        // -----------------------------------------

        /*
        Define one possible parameter set that ensures 100% validity
        (but does not account for triples and winning).

        Ensure that, given this parameter set, the network actually
        does return the correct result for all final-turn game board states.
        */
        TEST_METHOD(TicTacToeTrainer_proofOfConcept)
        {
            using namespace NeuralNetwork;

            // setup network
            std::shared_ptr<NodeNetwork> network = std::make_shared<NodeNetwork>();

            NetworkSizeData sizeData;
            sizeData.numInputNodes = 18;
            sizeData.numOutputNodes = 9;

            Assert::AreEqual(true, network->createNetwork(sizeData));

            // setup optimal parameter set:
            // - set weights for the edges connecting input and output nodes for the same board index to -1
            // - set everything else (remaining edges and node biases) to 0
            // ==> the empty cell (-1, -1) gets a positive value
            // ==> occupied cells (1, -1) get a zero value
            // ==> the empty cell has the highest value
            ParamSet pset;
            for (int n = 0; n < 9; n++)
            {
                for (int i = 0; i < 19; i++)
                {
                    if (n * 2 == i || n * 2 + 1 == i)
                    {
                        pset.params.push_back(-1);
                    }
                    else
                    {
                        pset.params.push_back(0);
                    }
                }
            }

            network->assignParameters(pset.params);

            AiPlayer aiPlayer(0, CellState::CS_PLAYER1, network);

            std::vector<std::vector<CellState>> gameStateCollection;
            TicTacToeLogic::collectInconclusiveFinalGameBoardStates(gameStateCollection);

            TicTacToeLogic logic;
            int countInvalid = 0;

            // for each possible permutation on inconclusive last-turn states,
            // try whether the ai makes a valid move
            for (const auto& gameCells : gameStateCollection)
            {
                const int nextMove = aiPlayer.decideMove(gameCells);
                logic.setGameCells(gameCells);
                if (!logic.isValidMove(0, nextMove))
                {
                    countInvalid++;
                }
            }

            Assert::AreEqual(0, countInvalid);
        }
    };
}
