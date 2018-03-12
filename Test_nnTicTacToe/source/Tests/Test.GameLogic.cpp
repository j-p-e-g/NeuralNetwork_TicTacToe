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

        // if everything is free, all cells get the same value
        TEST_METHOD(TicTacToeLogic_getExpectedOutput_allFree)
        {
            TicTacToeLogic ticTacToe;

            std::vector<double> expectedOutput;
            ticTacToe.getExpectedOutput(0, expectedOutput);

            Assert::AreEqual(9, static_cast<int>(expectedOutput.size()));
            for (unsigned int k = 1; k < expectedOutput.size(); k++)
            {
                Assert::AreEqual(expectedOutput[k-1], expectedOutput[k], 0.0001);
            }
        }

        // if only one cell is free, that cell gets an output of 1
        // everything else gets 0
        // 1 _ 2
        // 2 1 1
        // 2 1 2
        TEST_METHOD(TicTacToeLogic_getExpectedOutput_oneFree)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(0, 0);
            ticTacToe.applyMove(1, 2);
            ticTacToe.applyMove(1, 3);
            ticTacToe.applyMove(0, 4);
            ticTacToe.applyMove(1, 5);
            ticTacToe.applyMove(1, 6);
            ticTacToe.applyMove(0, 7);
            ticTacToe.applyMove(1, 8);

            const std::vector<double> expected = { 0, 1, 0, 0, 0, 0, 0, 0, 0 };
            std::vector<double> expectedOutput;
            ticTacToe.getExpectedOutput(0, expectedOutput);

            Assert::AreEqual(expected.size(), expectedOutput.size());
            for (unsigned int k = 0; k < expected.size(); k++)
            {
                Assert::AreEqual(expected[k], expectedOutput[k], 0.0001);
            }
        }

        // in general, occupied cells get zero
        // and cells offering a potential win or loss prevention get a higher value
        // 2 1 _
        // _ 1 1
        // 2 _ _
        TEST_METHOD(TicTacToeLogic_getExpectedOutput_general)
        {
            TicTacToeLogic ticTacToe;
            ticTacToe.applyMove(1, 0);
            ticTacToe.applyMove(0, 1);
            ticTacToe.applyMove(0, 4);
            ticTacToe.applyMove(0, 5);
            ticTacToe.applyMove(1, 6);

            std::vector<double> expectedOutput;
            ticTacToe.getExpectedOutput(1, expectedOutput);

            Assert::AreEqual(9, static_cast<int>(expectedOutput.size()));
            Assert::AreEqual(0, expectedOutput[0], 0.0001);
            Assert::AreEqual(0, expectedOutput[1], 0.0001);
            Assert::AreEqual(0, expectedOutput[4], 0.0001);
            Assert::AreEqual(0, expectedOutput[5], 0.0001);
            Assert::AreEqual(0, expectedOutput[6], 0.0001);

            // these two have the same priority and may not be worse than occupied cells
            Assert::AreEqual(expectedOutput[2], expectedOutput[8], 0.0001);
            Assert::AreEqual(true, expectedOutput[2] >= 0);

            // whether win/loss prevention get the same or different values
            // depends on the implementation but they get a better value than
            // a normal free cell
            Assert::AreEqual(true, expectedOutput[3] > expectedOutput[2]);
            Assert::AreEqual(true, expectedOutput[7] > expectedOutput[2]);
        }

        TEST_METHOD(TicTacToeLogic_getTripleCandidate_row)
        {
            // if a single triple can be completed, the completing cell will be returned
            // 1 _ 1
            // 2 2 _
            // _ 1 2

            std::vector<CellState> gameCells;
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER2);

            Assert::AreEqual(1, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER1, 0, 1, 2));
            Assert::AreEqual(-1, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER1, 3, 4, 5));
            Assert::AreEqual(-1, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER1, 6, 7, 8));

            Assert::AreEqual(-1, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER2, 0, 1, 2));
            Assert::AreEqual(5, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER2, 3, 4, 5));
            Assert::AreEqual(-1, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER2, 6, 7, 8));
        }

        TEST_METHOD(TicTacToeLogic_getTripleCandidate_column)
        {
            // if a single triple can be completed, the completing cell will be returned
            // _ 1 1
            // 2 1 2
            // 2 _ 2

            std::vector<CellState> gameCells;
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER2);

            Assert::AreEqual(-1, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER1, 0, 3, 6));
            Assert::AreEqual(7, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER1, 1, 4, 7));
            Assert::AreEqual(-1, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER1, 2, 5, 8));

            Assert::AreEqual(0, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER2, 0, 3, 6));
            Assert::AreEqual(-1, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER2, 1, 4, 7));
            Assert::AreEqual(-1, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER2, 2, 5, 8));
        }

        TEST_METHOD(TicTacToeLogic_getTripleCandidate_diagonal)
        {
            // if a single triple can be completed, the completing cell will be returned
            // 2 1 _
            // 2 1 2
            // 1 _ 2

            std::vector<CellState> gameCells;
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER2);

            Assert::AreEqual(-1, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER1, 0, 4, 8));
            Assert::AreEqual(2, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER1, 2, 4, 6));

            Assert::AreEqual(-1, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER2, 0, 4, 8));
            Assert::AreEqual(-1, TicTacToeLogic::getTripleCandidate(gameCells, CS_PLAYER2, 2, 4, 6));
        }

        TEST_METHOD(TicTacToeLogic_getTripleCandidates)
        {
            // collect triple candidates
            // _ 1 1
            // 2 1 _
            // 2 _ 1

            std::vector<CellState> gameCells;
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_PLAYER1);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER2);
            gameCells.push_back(CellState::CS_EMPTY);
            gameCells.push_back(CellState::CS_PLAYER1);

            {
                std::vector<int> candidates;
                TicTacToeLogic::getTripleCandidates(gameCells, CellState::CS_PLAYER1, candidates);
                Assert::AreEqual(4, static_cast<int>(candidates.size()));

                Assert::AreEqual(true, std::find(candidates.begin(), candidates.end(), 0) != candidates.end());
                Assert::AreEqual(true, std::find(candidates.begin(), candidates.end(), 5) != candidates.end());
                Assert::AreEqual(true, std::find(candidates.begin(), candidates.end(), 7) != candidates.end());
            }

            {
                std::vector<int> candidates;
                TicTacToeLogic::getTripleCandidates(gameCells, CellState::CS_PLAYER2, candidates);
                Assert::AreEqual(1, static_cast<int>(candidates.size()));
                Assert::AreEqual(true, std::find(candidates.begin(), candidates.end(), 0) != candidates.end());
            }
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
