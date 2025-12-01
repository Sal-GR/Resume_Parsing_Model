# Resume_Parsing_Model

Run generate_adversarial.py --> data/adversarial/adversarial_resumes.csv

Run evaluate_clean_resumes.py --> data/clean_parser_output.csv

run evaluate_adversarial.py --> data/adversarial/adversarial_parser_output.csv, [ALREADY IN REPO: data/adversarial/attack_summary.csv]

Run extract_features.py --> data/features/resume_features.csv [CREATE FOLDER: data/features]

Run train_adveserial_detector.py --> models/rf_adversarial_detector.pkl, [ALREADY IN REPO: models/feature_importances.csv], [ALREADY IN REPO: models/logreg_adversarial_detector.pkl], [ALREADY IN REPO: models/scaler.pkl]
