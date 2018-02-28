#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

# example:
#  Rscript.exe scripts\createAllSetsPlot.r data\iteration_scores data\best_iteration_stats.csv

if (length(args)==0)
{
    stop("At least one argument must be supplied (input file).\nCall with \"Rscript.exe scriptName inputFileWithoutExtension [optionalOtherInputFile]\"", call.=FALSE)
}

inputFile = paste(args[1], "csv", sep=".")
outputFile = paste(args[1], "png", sep=".")
paste("Input file: ", inputFile)
paste("Output file: ", outputFile)

data <- read.table(inputFile, header=TRUE, sep=",")

png(filename=outputFile,width=800,height=720)

boxplot(data$Score~data$Iteration, main="Score progression during training", xlab="Iterations", ylab="Score", cex.main = 1.75, cex.lab=1.3)

miny = 0
maxy = max(data$Score, na.rm = TRUE)
axis(side=2, at=seq(miny, maxy + 10, by=10))

abline(lm(data$Score~data$Iteration), col="red")

if (length(args) > 1)
{
    secondInputFile = args[2]
    if (!grepl("csv", secondInputFile))
    {
        secondInputFile = paste(secondInputFile, "csv", sep=".")
    }
    
    scoreData <- read.table(secondInputFile, header=TRUE, sep=",")
    lines(scoreData$Iteration, scoreData$Score, type='s', col="green")
    abline(lm(scoreData$Score~scoreData$Iteration), col="blue")

    legend(x="topleft", legend=c("Predicted avg. score", "Best score", "Predicted best score"), col=c("red", "green", "blue"), cex=1.0, lty=c(1,1,1), xpd=TRUE)
}

dev.off()
