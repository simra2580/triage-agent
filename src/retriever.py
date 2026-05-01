def retrieve_answer(issue, corpus_df):
    for _, row in corpus_df.iterrows():
        content = str(row["content"]).lower()
        if any(word in content for word in issue.split()):
            return content
    return None