import pandas as pd
import argparse
import os

def prepare(csv_file_path, final_csv_file_path):
  df = pd.read_csv(csv_file_path)
  
  final_df = pd.DataFrame()
  final_df['Survived'] = df['Survived']
  final_df['Pclass'] = df['Pclass']
  final_df['Sex-Male'] = (df['Sex'] == 'male').astype(int)
  final_df['Age'] = df['Age'].fillna(df['Age'].mean())
  final_df['SibSp'] = df['SibSp']
  final_df['Parch'] = df['Parch']
  final_df['Fare'] = df['Fare']
  final_df['Cabin-A'] = df['Cabin'].str.contains('A', na=False).astype(int)
  final_df['Cabin-B'] = df['Cabin'].str.contains('B', na=False).astype(int)
  final_df['Cabin-C'] = df['Cabin'].str.contains('C', na=False).astype(int)
  final_df['Cabin-D'] = df['Cabin'].str.contains('D', na=False).astype(int)
  final_df['Cabin-E'] = df['Cabin'].str.contains('E', na=False).astype(int)
  final_df['Cabin-F'] = df['Cabin'].str.contains('F', na=False).astype(int)
  final_df['Cabin-G'] = df['Cabin'].str.contains('G', na=False).astype(int)
  final_df['Cabin_Count'] = df['Cabin'].fillna('').str.count('\s').astype(int)
  final_df['Embarked-C'] = (df['Embarked'] == 'C').astype(int)
  final_df['Embarked-Q'] = (df['Embarked'] == 'Q').astype(int)
  final_df['Embarked-S'] = (df['Embarked'] == 'S').astype(int)
  
  final_df.to_csv(final_csv_file_path, index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('in_csv_file')
  parser.add_argument('out_csv_file')
  args = parser.parse_args()

  prepare(args.in_csv_file, args.out_csv_file)