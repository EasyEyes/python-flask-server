# import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Ingest audio samples from client.")
parser.add_argument("--sampleRate", type=int, help="Sample rate of payload, in hz")
parser.add_argument("--payload", nargs="+", type=float, help="Array of audio samples")
parser.add_argument("--debug", action="store_true", help="Print debug info")


# def saveBufferToCSV(buffer, pathToFile):
#     df = pd.DataFrame(buffer)
#     df.to_csv(pathToFile)


# def readCSVData(pathToFile):
#     df = pd.read_csv(pathToFile)
#     df.dropna(inplace=True)
#     return df.to_numpy()[:, 1]