import pandas as pd

class StudentData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def to_dataframe(self):
        return pd.DataFrame([{
            "gender": self.gender,
            "race/ethnicity": self.race_ethnicity,
            "parental level of education": self.parental_level_of_education,
            "lunch": self.lunch,
            "test preparation course": self.test_preparation_course,
            "reading score": self.reading_score,
            "writing score": self.writing_score
        }])
