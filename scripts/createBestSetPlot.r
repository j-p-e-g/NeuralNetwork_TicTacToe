#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

# example:
#  Rscript.exe scripts\createBestSetsPlot.r data\best_iteration_stats

if (length(args)==0)
{
    stop("At least one argument must be supplied (input file).\nCall with \"Rscript.exe scriptName inputFileWithoutExtension\"", call.=FALSE)
}

inputFile = paste(args[1], "csv", sep=".")
outputFile = paste(args[1], "png", sep=".")
paste("Input file: ", inputFile)
paste("Output file: ", outputFile)

data <- read.table(inputFile, header=TRUE, sep=",")

png(filename=outputFile)

total = data$CountInvalid[1] + data$CountLost[1] + data$CountTied[1] + data$CountWon[1]
paste("Total: ", total)

miny = 0
maxy = total

plot(data$Iteration, data$CountInvalid, main="Game end state progression during training", xlab="Iterations", ylab="Count", ylim=c(miny, maxy), type="l", col="red", cex.main = 1.75, cex.lab=1.3)
lines(data$Iteration, data$CountLost, type='s', col="blue")
lines(data$Iteration, data$CountTied, type='s', col="black")
lines(data$Iteration, data$CountWon, type='s', col="green")

legend(80, 120, legend=c("#Invalid", "#Lost", "#Tied", "#Won"), col=c("red", "blue", "black", "green"), cex=0.95, lty=c(1,1,1,1), xpd=TRUE)

dev.off()
