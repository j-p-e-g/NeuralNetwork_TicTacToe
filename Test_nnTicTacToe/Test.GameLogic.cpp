#include "stdafx.h"
#include "CppUnitTest.h"
#include "Game/GameLogic.h"

#include <set>

namespace GameLogicTest
{
    using namespace Microsoft::VisualStudio::CppUnitTestFramework;
    using namespace Game;

    TEST_CLASS(GameLogic_Test)
    {
    public:
        // -----------------------------------------
        // TicTacToeLogic
        // -----------------------------------------
        TEST_METHOD(TicTacToeLogic_default)
        {
            TicTacToeLogic ticTacToe;
            Assert::AreEqual(9, ticTacToe.getBoardSize());

            for (int x = 0; x < 2; x++)
            {
                for (int y = 0; y < 2; y++)
                {
                    Assert::AreEqual(true, CellState::CS_EMPTY == ticTacToe.getCellValue(x, y));
                }
            }
        }

        TEST_METHOD(TicTacToeLogic_applyMove)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(1, 8);
            ticTacToe.applyMove(0, 3);

            Assert::AreEqual(true, CellState::CS_PLAYER2 == ticTacToe.getCellValue(2, 2));
            Assert::AreEqual(true, CellState::CS_PLAYER1 == ticTacToe.getCellValue(1, 0));
            Assert::AreEqual(true, CellState::CS_EMPTY == ticTacToe.getCellValue(2, 1));
        }

        TEST_METHOD(TicTacToeLogic_isValidMove_default)
        {
            TicTacToeLogic ticTacToe;
            Assert::AreEqual(false, ticTacToe.isValidMove(0, -1));
            Assert::AreEqual(false, ticTacToe.isValidMove(0, 9));

            for (int k = 0; k < 9; k++)
            {
                Assert::AreEqual(true, ticTacToe.isValidMove(0, k));
            }
        }

        TEST_METHOD(TicTacToeLogic_isValidMove)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(0, 4);

            Assert::AreEqual(false, ticTacToe.isValidMove(0, 4));
            Assert::AreEqual(true, ticTacToe.isValidMove(0, 5));
        }

        TEST_METHOD(TicTacToeLogic_countCellState_default)
        {
            TicTacToeLogic ticTacToe;
            Assert::AreEqual(9, ticTacToe.countCellState(CS_EMPTY));
            Assert::AreEqual(0, ticTacToe.countCellState(CS_PLAYER1));
            Assert::AreEqual(0, ticTacToe.countCellState(CS_PLAYER2));
        }

        TEST_METHOD(TicTacToeLogic_countCellState)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(1, 0);
            ticTacToe.applyMove(1, 1);
            ticTacToe.applyMove(0, 5);

            Assert::AreEqual(6, ticTacToe.countCellState(CS_EMPTY));
            Assert::AreEqual(1, ticTacToe.countCellState(CS_PLAYER1));
            Assert::AreEqual(2, ticTacToe.countCellState(CS_PLAYER2));
        }

        TEST_METHOD(TicTacToeLogic_evaluateBoard_default)
        {
            TicTacToeLogic ticTacToe;
            Assert::AreEqual(true, GameState::GS_ONGOING == ticTacToe.evaluateBoard());
        }

        // the current player wins if their move completes a row, column or diagonal
        // entirely made up of cells flagged by the same player
        TEST_METHOD(TicTacToeLogic_evaluateBoard_row1)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(1, 0);
            ticTacToe.applyMove(1, 1);
            ticTacToe.applyMove(1, 2);

            Assert::AreEqual(true, GameState::GS_GAMEOVER_WON == ticTacToe.evaluateBoard());
        }

        TEST_METHOD(TicTacToeLogic_evaluateBoard_row2)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(1, 3);
            ticTacToe.applyMove(1, 4);
            ticTacToe.applyMove(1, 5);

            Assert::AreEqual(true, GameState::GS_GAMEOVER_WON == ticTacToe.evaluateBoard());
        }

        TEST_METHOD(TicTacToeLogic_evaluateBoard_row3)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(0, 6);
            ticTacToe.applyMove(0, 7);
            ticTacToe.applyMove(0, 8);

            Assert::AreEqual(true, GameState::GS_GAMEOVER_WON == ticTacToe.evaluateBoard());
        }

        TEST_METHOD(TicTacToeLogic_evaluateBoard_column1)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(0, 0);
            ticTacToe.applyMove(0, 3);
            ticTacToe.applyMove(0, 6);

            Assert::AreEqual(true, GameState::GS_GAMEOVER_WON == ticTacToe.evaluateBoard());
        }

        TEST_METHOD(TicTacToeLogic_evaluateBoard_column2)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(1, 1);
            ticTacToe.applyMove(1, 4);
            ticTacToe.applyMove(1, 7);

            Assert::AreEqual(true, GameState::GS_GAMEOVER_WON == ticTacToe.evaluateBoard());
        }

        TEST_METHOD(TicTacToeLogic_evaluateBoard_column3)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(0, 2);
            ticTacToe.applyMove(0, 5);
            ticTacToe.applyMove(0, 8);

            Assert::AreEqual(true, GameState::GS_GAMEOVER_WON == ticTacToe.evaluateBoard());
        }

        TEST_METHOD(TicTacToeLogic_evaluateBoard_diagonal1)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(0, 0);
            ticTacToe.applyMove(0, 4);
            ticTacToe.applyMove(0, 8);

            Assert::AreEqual(true, GameState::GS_GAMEOVER_WON == ticTacToe.evaluateBoard());
        }

        TEST_METHOD(TicTacToeLogic_evaluateBoard_diagonal2)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(1, 2);
            ticTacToe.applyMove(1, 4);
            ticTacToe.applyMove(1, 6);

            Assert::AreEqual(true, GameState::GS_GAMEOVER_WON == ticTacToe.evaluateBoard());
        }

        // ... but not if they're flagged by different players
        TEST_METHOD(TicTacToeLogic_evaluateBoard_blockedRow)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(1, 6);
            ticTacToe.applyMove(0, 7);
            ticTacToe.applyMove(0, 8);

            Assert::AreEqual(true, GameState::GS_ONGOING == ticTacToe.evaluateBoard());
        }

        TEST_METHOD(TicTacToeLogic_evaluateBoard_blockedColumn)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(1, 0);
            ticTacToe.applyMove(1, 3);
            ticTacToe.applyMove(0, 6);

            Assert::AreEqual(true, GameState::GS_ONGOING == ticTacToe.evaluateBoard());
        }

        TEST_METHOD(TicTacToeLogic_evaluateBoard_diagonalBlocked)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(0, 0);
            ticTacToe.applyMove(1, 4);
            ticTacToe.applyMove(0, 8);

            Assert::AreEqual(true, GameState::GS_ONGOING == ticTacToe.evaluateBoard());
        }

        // if the board is filled without any player having won, the game is over
        // oxx
        // xoo
        // oox
        TEST_METHOD(TicTacToeLogic_evaluateBoard_boardFilled)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(0, 0);
            ticTacToe.applyMove(1, 1);
            ticTacToe.applyMove(1, 2);
            ticTacToe.applyMove(1, 3);
            ticTacToe.applyMove(0, 4);
            ticTacToe.applyMove(0, 5);
            ticTacToe.applyMove(0, 6);
            ticTacToe.applyMove(0, 7);
            ticTacToe.applyMove(1, 8);

            Assert::AreEqual(true, GameState::GS_GAMEOVER_TIMEOUT == ticTacToe.evaluateBoard());
        }

        // otherwise, if no rows are formed, the game continues
        TEST_METHOD(TicTacToeLogic_evaluateBoard_scattered)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(0, 0);
            ticTacToe.applyMove(1, 1);
            ticTacToe.applyMove(0, 4);
            ticTacToe.applyMove(0, 6);
            ticTacToe.applyMove(1, 8);
            Assert::AreEqual(true, GameState::GS_ONGOING == ticTacToe.evaluateBoard());
        }

        TEST_METHOD(TicTacToeLogic_collectInconclusiveFinalGameBoardStateInputValues)
        {
            std::vector<std::vector<CellState>> collectedGameStates;

            TicTacToeLogic::collectInconclusiveFinalGameBoardStates(collectedGameStates);

            // 630 permutations in total, of these 408 are already concluded by one or the other player winning
            Assert::AreEqual(222, static_cast<int>(collectedGameStates.size()));

            std::set<std::vector<CellState>> uniqueCollection;
            for (const auto& list : collectedGameStates)
            {
                uniqueCollection.emplace(list);
                Assert::AreEqual(9, static_cast<int>(list.size()));

                int countEmpty = 0;
                int countPlayerOne = 0;
                int countPlayerTwo = 0;
                for (const auto& val : list)
                {
                    switch (val)
                    {
                    case CellState::CS_EMPTY:
                        countEmpty++;
                        break;
                    case CellState::CS_PLAYER1:
                        countPlayerOne++;
                        break;
                    case CellState::CS_PLAYER2:
                        countPlayerTwo++;
                        break;
                    }
                }

                // 1 empty cell, 4 occupied cells for each player
                Assert::AreEqual(1, countEmpty);
                Assert::AreEqual(4, countPlayerOne);
                Assert::AreEqual(4, countPlayerTwo);
            }

            // no duplicates
            Assert::AreEqual(collectedGameStates.size(), uniqueCollection.size());
        }
    };
}
