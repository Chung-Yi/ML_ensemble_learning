import os
from models.stacking import StackingClassifier

def main():

    stacking = StackingClassifier()

    stacking.load_data()
    stacking.classifier()

    print(stacking.acc)

if __name__ == "__main__":
    main()