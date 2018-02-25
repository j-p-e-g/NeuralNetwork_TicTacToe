#include "stdafx.h"
#include "CppUnitTest.h"
#include "Game/Player.h"
#include "NeuralNetwork/NodeNetwork.h"

namespace PlayerTest
{
    using namespace Microsoft::VisualStudio::CppUnitTestFramework;
    using namespace Game;

    TEST_CLASS(Player_Test)
    {
    public:
        // -----------------------------------------
        // RandomPlayer
        // -----------------------------------------
        TEST_METHOD(RandomPlayer_decideMove)
        {
            // if only one cell is free, that cell index will be returned
            std::vector<CellState> gameCells;
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_EMPTY); // index 5
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER1);

            RandomPlayer player(1);
            const int result = player.decideMove(gameCells);
            Assert::AreEqual(5, result);
        }

        // -----------------------------------------
        // AiPlayer
        // -----------------------------------------
        TEST_METHOD(AiPlayer_decideMove)
        {
            using namespace NeuralNetwork;

            std::vector<CellState> gameCells;
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER2);

            const std::vector<double> expectedInputValues =
            {
               -1, -1, // empty
               1, -1,  // player1
               1, -1,  // player1
               -1, -1, // empty
               -1, 1,  // player2
               -1, -1, // empty
               -1, -1, // empty
               -1, -1, // empty
               -1, 1  // player2
            };

            std::shared_ptr<NodeNetwork> network = std::make_shared<NodeNetwork>();
            AiPlayer player(0, network);

            std::vector<double> inputValues;
            player.getNodeNetworkInputValues(gameCells, inputValues);
            Assert::AreEqual(18, static_cast<int>(inputValues.size()));

            for (unsigned int k = 0; k < inputValues.size(); k++)
            {
                Assert::AreEqual(expectedInputValues[k], inputValues[k], 0.0001);
            }
        }
    };
}
