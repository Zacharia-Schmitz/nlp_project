import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import pickle
import wrangle as w
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter
import random
import joblib
import matplotlib.pyplot as plt

nltk.download("stopwords")


def fetch_github_data(url):
    """
    Fetch data from a GitHub API endpoint.

    Parameters:
    - url (str): The URL of the API endpoint.

    Returns:
    - dict: The JSON response from the API.
    """

    # Define the base URL for the GitHub search API
    BASE_URL = "https://github.com/search?q=stars%3A%3E0+language%3A{language}&type=repositories&l={language}&p={page}"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        print(
            f"Status code: {response.status_code} - Rate limit exceeded. Ending script."
        )
        sys.exit()
    else:
        print(f"Status code: {response.status_code} - Failed to fetch data from {url}.")
        return None


def extract_repo_links(data, processed_repos):
    """
    Extract repository links from the JSON response of a GitHub API endpoint.

    Parameters:
    - data (dict): The JSON response from the API.
    - processed_repos (set): A set of repository URLs that have already been processed.

    Returns:
    - list: A list of repository URLs that have not been processed before.
    """
    repo_links = []
    for result in data["payload"]["results"]:
        owner_login = result["repo"]["repository"]["owner_login"]
        repo_name = result["repo"]["repository"]["name"]
        repo_url = f"/{owner_login}/{repo_name}"
        if repo_url not in processed_repos:
            repo_links.append(repo_url)
            processed_repos.add(repo_url)
    return repo_links


def get_readme_content(repo):
    """
    Get the content of the README file for a repository.

    Parameters:
    - repo (str): The URL of the repository.

    Returns:
    - str: The content of the README file, or None if the file could not be found.
    """
    readme_url = "https://github.com" + repo + "/blob/master/README.md"
    response = requests.get(readme_url)
    soup = BeautifulSoup(response.content, "html.parser")
    readme_content = soup.select_one("article")
    return readme_content.get_text() if readme_content else None


def fetch_readmes(
    languages=["python"], num_repos=5, start_page=1, sleep_time=1, verbose=True
):
    """
    Fetch README files from GitHub repositories.

    Parameters:
    - languages (list): A list of programming languages to search for.
    - num_repos (int): The number of repositories to fetch README files from.
    - start_page (int): The page number to start searching from.
    - sleep_time (int): The number of seconds to wait between requests.
    - verbose (bool): Whether to print progress messages.

    Returns:
    - pandas.DataFrame: A DataFrame containing the language, repository URL, and README content for each fetched README file.
    """
    all_readmes = []
    processed_repos = set()

    for language in languages:
        if verbose:
            print(f"Processing {language} repositories...")
        page = start_page
        total_processed = 0

        while total_processed < num_repos:
            url = BASE_URL.format(language=language, page=page)
            data = fetch_github_data(url)
            if not data:
                print(f"Failed to fetch data for {language} on page {page}.")
                break

            repo_links = extract_repo_links(data, processed_repos)
            for repo in repo_links:
                if total_processed >= num_repos:
                    break
                readme_content = get_readme_content(repo)
                if readme_content:
                    all_readmes.append((language, repo, readme_content))
                    total_processed += 1
                    if verbose:
                        print(
                            f"Fetched README {total_processed} of {num_repos} for {language}."
                        )
                time.sleep(sleep_time)  # Add a delay between requests
            page += 1

        if num_repos > 10 and page <= num_repos // 10:
            if verbose:
                print(f"Fetching additional pages for {language}...")
            for i in range(page, num_repos // 10 + 1):
                url = BASE_URL.format(language=language, page=i)
                data = fetch_github_data(url)
                if not data:
                    print(f"Failed to fetch data for {language} on page {i}.")
                    break

                repo_links = extract_repo_links(data, processed_repos)
                for repo in repo_links:
                    if total_processed >= num_repos:
                        break
                    readme_content = get_readme_content(repo)
                    if readme_content:
                        all_readmes.append((language, repo, readme_content))
                        total_processed += 1
                        if verbose:
                            print(
                                f"Fetched README {total_processed} of {num_repos} for {language}."
                            )
                    time.sleep(sleep_time)  # Add a delay between requests

            if verbose:
                print(
                    f"Finished processing {language} repositories. Fetched a total of {total_processed} READMEs."
                )

    df = pd.DataFrame(all_readmes, columns=["language", "repo", "readme"])
    if verbose:
        print(f"Total README Count: {len(df)}")
    return df
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe.
    """

    dataframeinfo = []

    # Check information about the index
    index_dtype = DataFrame.index.dtype
    index_unique_vals = DataFrame.index.unique()
    index_num_unique = DataFrame.index.nunique()
    index_num_null = DataFrame.index.isna().sum()
    index_pct_null = index_num_null / len(DataFrame.index)

    if pd.api.types.is_numeric_dtype(index_dtype) and not isinstance(
        DataFrame.index, pd.RangeIndex
    ):
        index_min_val = DataFrame.index.min()
        index_max_val = DataFrame.index.max()
        index_range_vals = (index_min_val, index_max_val)
    elif pd.api.types.is_datetime64_any_dtype(index_dtype):
        index_min_val = DataFrame.index.min()
        index_max_val = DataFrame.index.max()
        index_range_vals = (
            index_min_val.strftime("%Y-%m-%d"),
            index_max_val.strftime("%Y-%m-%d"),
        )

        # Check for missing dates in the index if dates kwarg is True
        if dates:
            full_date_range = pd.date_range(
                start=index_min_val, end=index_max_val, freq="D"
            )
            missing_dates = full_date_range.difference(DataFrame.index)
            if not missing_dates.empty:
                print(
                    f"Missing dates in index: ({len(missing_dates)} Total) {missing_dates.tolist()}"
                )
    else:
        index_range_vals = None

    dataframeinfo.append(
        [
            "index",
            index_dtype,
            index_num_unique,
            index_num_null,
            index_pct_null,
            index_unique_vals,
            index_range_vals,
        ]
    )

    print(f"Total rows: {DataFrame.shape[0]}")
    print(f"Total columns: {DataFrame.shape[1]}")

    if reports:
        describe = DataFrame.describe().round(2)
        print(describe)

    if graphs:
        DataFrame.hist(figsize=(10, 10))
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    for column in DataFrame.columns:
        dtype = DataFrame[column].dtype
        unique_vals = DataFrame[column].unique()
        num_unique = DataFrame[column].nunique()
        num_null = DataFrame[column].isna().sum()
        pct_null = DataFrame[column].isna().mean().round(5)

        if pd.api.types.is_numeric_dtype(dtype):
            min_val = DataFrame[column].min()
            max_val = DataFrame[column].max()
            mean_val = DataFrame[column].mean()
            range_vals = (min_val, max_val, mean_val)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            min_val = DataFrame[column].min()
            max_val = DataFrame[column].max()
            range_vals = (min_val.strftime("%Y-%m-%d"), max_val.strftime("%Y-%m-%d"))

            if dates:
                full_date_range_col = pd.date_range(
                    start=min_val, end=max_val, freq="D"
                )
                missing_dates_col = full_date_range_col.difference(DataFrame[column])
                if not missing_dates_col.empty:
                    print(
                        f"Missing dates in column '{column}': ({len(missing_dates_col)} Total) {missing_dates_col.tolist()}"
                    )
                else:
                    print(f"No missing dates in column '{column}'")

        else:
            range_vals = None

        dataframeinfo.append(
            [column, dtype, num_unique, num_null, pct_null, unique_vals, range_vals]
        )

    return pd.DataFrame(
        dataframeinfo,
        columns=[
            "col_name",
            "dtype",
            "num_unique",
            "num_null",
            "pct_null",
            "unique_values",
            "range (min, max, mean)",
        ],
    )


def preprocess_text(text):
    """
    Preprocesses a text string by removing unwanted characters, converting to lowercase, and removing stopwords.

    Parameters:
    - text (str): The text string to preprocess.

    Returns:
    - str: The preprocessed text string.
    """
    # Remove all newline characters
    text = re.sub(r"\\n", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Replace hyphens with spaces
    text = re.sub(r"-", " ", text)
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Convert to lowercase
    text = text.lower()
    # Remove extra white spaces
    text = " ".join(text.split())

    # Load the list of stopwords
    stop_words = set(stopwords.words("english"))

    # Add custom stopwords
    stop_words.update(["use", "using", "used", "code", "codes", "file"])

    # Tokenize the text and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def get_common_words(df):
    """
    Returns a DataFrame containing the 5 most common words for each programming language in the input DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the preprocessed README files and their corresponding programming languages.

    Returns:
    - pandas.DataFrame: A DataFrame containing the 5 most common words for each programming language and their respective counts.
    """
    # Tokenize the preprocessed READMEs and count the occurrences of each word
    word_counts = {}
    for language in df["language"].unique():
        language_readmes = df[df["language"] == language]["preprocessed_readme"]
        language_word_counts = Counter(" ".join(language_readmes).split())
        word_counts[language] = language_word_counts

    # Get the 5 most common words for each language
    common_words = {}
    for language, counts in word_counts.items():
        common_words[language] = counts.most_common(5)

    # Convert the results to a DataFrame for better visualization
    common_words_df = pd.DataFrame(columns=["Language", "Word", "Count"])
    for language, words in common_words.items():
        for i, (word, count) in enumerate(words):
            common_words_df = pd.concat(
                [
                    common_words_df,
                    pd.DataFrame(
                        {"Language": language, "Word": word, "Count": count}, index=[0]
                    ),
                ]
            )

    return common_words_df


def plot_common_words(common_words):
    """
    Plots a bar chart of the 5 most common words for each programming language in the input dictionary.

    Parameters:
    - common_words (dict): A dictionary containing the 5 most common words for each programming language.

    Returns:
    - None
    """
    # Check that the input dictionary is formatted correctly
    if not isinstance(common_words, dict):
        raise TypeError("Input must be a dictionary")

    # Create a separate plot for each language
    for language, words_counts in common_words.items():
        # Check that the value for each language is a list of tuples
        if not isinstance(words_counts, list) or not all(
            isinstance(x, tuple) and len(x) == 2 for x in words_counts
        ):
            raise ValueError(f"Value for language '{language}' is not a list of tuples")

        # Get the top 5 most common words for the language
        words, counts = zip(*words_counts)
        words = list(words)
        counts = list(counts)

        # Create a bar chart of the top 5 most common words for the language
        plt.figure(figsize=(10, 6))
        plt.bar(words, counts)
        plt.title(f"Top 5 Most Common Words in {language} READMEs")
        plt.xlabel("Word")
        plt.ylabel("Count")
        plt.xticks(rotation=90)
        plt.show()


def plot_readme_length_distribution(df):
    """
    Plots the distribution of README lengths by programming language in the input DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the preprocessed README files and their corresponding programming languages.

    Returns:
    - None
    """
    # Calculate the length of each README
    df["readme_length"] = df["preprocessed_readme"].apply(len)

    # Plot the distribution of README lengths by programming language
    plt.figure(figsize=(15, 8))
    for lang in df["language"].unique():
        subset = df[df["language"] == lang]
        plt.hist(subset["readme_length"], bins=50, alpha=0.6, label=lang)

    plt.title("Distribution of README Lengths by Programming Language")
    plt.xlabel("README Length")
    plt.ylabel("Number of Repositories")
    plt.legend()
    plt.grid(axis="y")
    plt.show()


def predict_language(
    readme_string=None,
    preprocess_func=preprocess_text,
    tfidf_path="support_files/tfidf_vectorizer.pkl",
    logreg_path="support_files/logreg_model.pkl",
):
    """
    Predict the language of a readme string.

    Parameters:
    - readme_string (str): The input README string. If None, a random string will be selected from the CSV file.
    - preprocess_func (function): The function to preprocess the input text.
    - tfidf_path (str): Path to the saved TfidfVectorizer .pkl file.
    - logreg_path (str): Path to the saved LogisticRegression model .pkl file.

    Returns:
    - str: Predicted language label.
    """

    # If no readme_string is provided, select a random one from the CSV file
    if readme_string is None:
        df = pd.read_csv("support_files/all_readmes_processed.csv")
        random_index = random.randint(0, len(df) - 1)
        readme_string = df.loc[random_index, "preprocessed_readme"]
        print(f"Random String: {readme_string}\n")
        print(f"Actual Language: {df.loc[random_index, 'language']}")

    # Preprocess the input string
    preprocessed_string = preprocess_func(readme_string)

    # Load the TfidfVectorizer and transform the string
    tfidf_vectorizer = joblib.load(tfidf_path)
    transformed_string = tfidf_vectorizer.transform([preprocessed_string])

    # Load the LogisticRegression model and predict
    logreg_model = joblib.load(logreg_path)
    prediction = logreg_model.predict(transformed_string)

    # Return the predicted language
    print(f"Predicted: {prediction[0]}")


def plot_unique_identifier_words(df, threshold=0, n=10):
    """
    Plots a horizontal bar chart of the number of unique identifier words for each programming language in the input DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the preprocessed README files and their corresponding programming languages.
    - threshold (float): The threshold frequency for a word to be considered unique to a language. Default is 0.
    - n (int): The number of top words to consider for each language. Default is 10.

    Returns:
    - None
    """

    def top_words_for_language(lang, n=10):
        """Get the top n words for a given language."""
        text = " ".join(df[df["language"] == lang]["preprocessed_readme"])
        word_counts = Counter(text.split())
        return word_counts.most_common(n)

    def is_word_unique_to_language(word, lang, threshold=0):
        """Check if a word is unique to a given language based on a threshold frequency in other languages."""
        total_occurrences = sum(
            [1 for readme in df["preprocessed_readme"] if word in readme]
        )
        lang_occurrences = sum(
            [
                1
                for readme in df[df["language"] == lang]["preprocessed_readme"]
                if word in readme
            ]
        )

        # If the word occurs predominantly in the given language, it's considered unique
        return (lang_occurrences / total_occurrences) > threshold

    unique_identifier_words = {}

    # For each language, identify potential unique identifier words
    for lang in df["language"].unique():
        top_words = [word[0] for word in top_words_for_language(lang, n)]
        unique_words = [
            word
            for word in top_words
            if is_word_unique_to_language(word, lang, threshold)
        ]
        unique_identifier_words[lang] = unique_words

    # Extract data for plotting
    languages = list(unique_identifier_words.keys())
    num_unique_words = [len(unique_identifier_words[lang]) for lang in languages]
    unique_words = [", ".join(unique_identifier_words[lang]) for lang in languages]

    # Create bar chart
    plt.figure(
        figsize=(10, len(languages) * 0.5)
    )  # Adjusting the height based on number of languages
    bars = plt.barh(languages, num_unique_words, color="skyblue")

    # Annotate bars with the unique words
    for bar, words in zip(bars, unique_words):
        plt.text(
            bar.get_width()
            - (
                0.02 * max(num_unique_words)
            ),  # Positioning the text a bit inside the bar's end
            bar.get_y() + bar.get_height() / 2,
            words,
            va="center",
            ha="right",
            color="black",
            fontsize=10,
        )

    plt.xlabel("Number of Unique Identifier Words")
    plt.ylabel("Languages")
    plt.title("Unique Identifier Words by Language")
    plt.tight_layout()
    plt.show()


def get_common_words(df, plot=True):
    """
    Returns a DataFrame containing the 5 most common words for each programming language in the input DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the preprocessed README files and their corresponding programming languages.
    - plot (bool): Whether to plot the results. Default is True.

    Returns:
    - pandas.DataFrame: A DataFrame containing the 5 most common words for each programming language and their respective counts.
    """
    # Tokenize the preprocessed READMEs and count the occurrences of each word
    word_counts = {}
    for language in df["language"].unique():
        language_readmes = df[df["language"] == language]["preprocessed_readme"]
        language_word_counts = Counter(" ".join(language_readmes).split())
        word_counts[language] = language_word_counts

    # Get the 5 most common words for each language
    common_words = {}
    for language, counts in word_counts.items():
        common_words[language] = counts.most_common(5)

    # Convert the results to a DataFrame for better visualization
    common_words_df = pd.DataFrame(columns=["Language", "Word", "Count"])
    for language, words in common_words.items():
        for i, (word, count) in enumerate(words):
            common_words_df = pd.concat(
                [
                    common_words_df,
                    pd.DataFrame(
                        {"Language": language, "Word": word, "Count": count}, index=[0]
                    ),
                ]
            )

    # Plot the results if requested
    if plot:
        for language in common_words.keys():
            words, counts = zip(*common_words[language])
            plt.figure(figsize=(10, 6))
            plt.bar(words, counts)
            plt.title(f"Top 5 Most Common Words in {language} READMEs")
            plt.xlabel("Word")
            plt.ylabel("Count")
            plt.xticks(rotation=90)
            plt.show()

    return common_words_df
