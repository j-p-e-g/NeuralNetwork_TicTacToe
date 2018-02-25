#include "stdafx.h"
#include "Game/TicTacToeTrainer.h"

int main()
{
    Game::TicTacToeTrainer trainer;
    if (trainer.setup())
    {
        trainer.run();
    }

    return 0;
}
