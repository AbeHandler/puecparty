from awslambda import get_longitudinal_scores

def test_busn_long_scores_grading():
	scores = get_longitudinal_scores(instructor_name="Handler, Abram")
	number = [o for o in scores if o["Year"] == 2024 and o["Metric"] == "Grading" and o['Instructor'] == 'ALL_BUSN'][0]["Score"]
	assert round(number, 2) == 4.23
	# to get this number by hand, look at 42.xlsx in fcq_processor.py and run 
	# =AVERAGEIFS(AD8:AD74250, D8:D74250, "BUSN", B8:B74250, 2024, AD8:AD74250, "<>0") in row 74251 col AD

def test_busn_long_scores_question():
	scores = get_longitudinal_scores(instructor_name="Handler, Abram")
	number = [o for o in scores if o["Year"] == 2024 and o["Metric"] == "Questions" and o['Instructor'] == 'ALL_BUSN'][0]["Score"]
	assert round(number, 2) == 4.5
	# to get this number by hand, look at 42.xlsx in fcq_processor.py and run 
	# =AVERAGEIFS(AD8:AD74250, D8:D74250, "BUSN", B8:B74250, 2024, AD8:AD74250, "<>0") in row 74251 col AE

def test_busn_long_scores_text():
	scores = get_longitudinal_scores(instructor_name="Handler, Abram")
	number = [o for o in scores if o["Year"] == 2024 and o["Metric"] == "Tech" and o['Instructor'] == 'ALL_BUSN'][0]["Score"]
	assert round(number, 2) == 4.43
	# to get this number by hand, look at 42.xlsx in fcq_processor.py and run 
	# =AVERAGEIFS(AD8:AD74250, D8:D74250, "BUSN", B8:B74250, 2024, AD8:AD74250, "<>0") in row 74251 col AF

