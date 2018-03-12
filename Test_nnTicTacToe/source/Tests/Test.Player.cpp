#include "stdafx.h"

#include <set>

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
        TEST_METHOD(RandomPlayer_decideMove_single)
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

            RandomPlayer player(17, CellState::CS_PLAYER1);
            const int result = player.decideMove(gameCells);
            Assert::AreEqual(5, result);
        }

        TEST_METHOD(RandomPlayer_decideMove_multi)
        {
            // only the unoccupied cells may be returned
            std::vector<CellState> gameCells;
            gameCells.push_back(CellState::CS_EMPTY); // index 0
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_EMPTY); // index 3
            gameCells.push_back(CellState::CS_EMPTY); // index 4
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_EMPTY); // index 7
            gameCells.push_back(CellState::CS_EMPTY); // index 8

            std::vector<int> expectedMoves = { 0, 3, 4, 7, 8 };

            RandomPlayer player(94, CellState::CS_PLAYER2);

            std::vector<int> randomMoves;
            std::set<int> uniqueRandomMoves;
            for (int k = 0; k < 10; k++)
            {
                const int result = player.decideMove(gameCells);
                randomMoves.push_back(result);
                uniqueRandomMoves.emplace(result);

                // always one of the empty cells
                Assert::AreEqual(true, std::find(expectedMoves.begin(), expectedMoves.end(), result) != expectedMoves.end());
            }

            // not all the same value (can happen but is unlikely)
            Assert::AreEqual(true, uniqueRandomMoves.size() > 1);
        }

        // -----------------------------------------
        // SemiRandomPlayer
        // -----------------------------------------
        TEST_METHOD(SemiRandomPlayer_decideMove_avoidLoss_row)
        {
            // if a single triple can be completed, the completing cell will be returned
            // 1 _ _
            // 2 2 _
            // _ 1 2

            std::vector<CellState> gameCells;
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER2);

            SemiRandomPlayer player(2, CellState::CS_PLAYER1);
            const int result = player.decideMove(gameCells);
            Assert::AreEqual(5, result);
        }

        TEST_METHOD(SemiRandomPlayer_decideMove_win_column)
        {
            // if multiple triples can be completed, the winning move takes precedence
            // _ _ 2
            // 1 1 2
            // 2 1 _

            std::vector<CellState> gameCells;
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_EMPTY);

            SemiRandomPlayer player(89, CellState::CS_PLAYER2);
            const int result = player.decideMove(gameCells);
            Assert::AreEqual(8, result);
        }

        TEST_METHOD(SemiRandomPlayer_decideMove_avoidLoss_diagonal)
        {
            // if a single triple can be completed, the completing cell will be returned
            // 1 2 2
            // 2 _ _
            // _ _ 1

            std::vector<CellState> gameCells;
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER1);

            SemiRandomPlayer player(1, CellState::CS_PLAYER2);
            const int result = player.decideMove(gameCells);
            Assert::AreEqual(4, result);
        }

        // -----------------------------------------
        // AiPlayer
        // -----------------------------------------
        TEST_METHOD(AiPlayer_getNodeNetworkInputValues_player1)
        {
            using namespace NeuralNetwork;

            std::shared_ptr<NodeNetwork> network = std::make_shared<NodeNetwork>();
            AiPlayer player(5, CellState::CS_PLAYER1, network);

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

            std::vector<double> inputValues;
            player.getNodeNetworkInputValues(gameCells, inputValues);
            Assert::AreEqual(18, static_cast<int>(inputValues.size()));

            for (unsigned int k = 0; k < inputValues.size(); k++)
            {
                Assert::AreEqual(expectedInputValues[k], inputValues[k], 0.0001);
            }
        }

        TEST_METHOD(AiPlayer_getNodeNetworkInputValues_player2)
        {
            using namespace NeuralNetwork;

            std::shared_ptr<NodeNetwork> network = std::make_shared<NodeNetwork>();
            AiPlayer player(5, CellState::CS_PLAYER2, network);

            // same setup
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

            // input switches player1 and player2
            const std::vector<double> expectedInputValues =
            {
                -1, -1, // empty
                -1, 1,  // player2
                -1, 1,  // player2
                -1, -1, // empty
                1, -1,  // player1
                -1, -1, // empty
                -1, -1, // empty
                -1, -1, // empty
                1, -1   // player1
            };

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
