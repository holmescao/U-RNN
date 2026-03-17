import logging
import yaml
import hashlib
import os
import pandas as pd
import wandb
import git


def git_code(commit_info=""):
    """
    Commit all changes in the current directory to the git repository with the provided message.
    Returns the commit hash of the new commit.

    Parameters:
    - commit_info: A string message to use for the commit.
    """
    repo_dir = os.getcwd()
    repo = git.Repo(repo_dir)
    repo.git.add(".")
    repo.git.commit("-m", commit_info)
    print("Successfully committed current version.")
    commit = repo.head.commit
    return commit.hexsha


def wandb_init(job_type, id, name, config, project, notes, wandb_dir):
    """
    Initialize a Weights and Biases run with the given parameters.

    Parameters:
    - job_type: Job type in Weights and Biases.
    - id: Unique identifier for the run.
    - name: Name of the run.
    - config: Configuration parameters for the run.
    - project: Project name in Weights and Biases.
    - notes: Additional notes for the run.
    - wandb_dir: Directory to store Weights and Biases related files.
    """
    run = wandb.init(
        job_type=job_type, id=id, name=name, project=project,
        config=config, notes=notes, dir=wandb_dir, save_code=True
    )
    return run


def init_logging(save_dir):
    """
    Initializes logging to a file in the specified directory.

    Parameters:
    - save_dir: Directory where the log file will be saved.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    filename = os.path.join(save_dir, "log.log")
    logging.basicConfig(filename=filename,
                        format="%(asctime)s - %(pathname)s[line:%(lineno)d]: %(message)s",
                        level=logging.INFO)
    return filename


def params_to_yaml(args, save_dir):
    """
    Save the given parameters to a YAML file in the specified directory.

    Parameters:
    - args: Arguments to be saved.
    - save_dir: Directory where the YAML file will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "exp_args.yaml")
    with open(save_path, "w") as outfile:
        yaml.dump(args, outfile)


def hash_string(string):
    """
    Returns the MD5 hash of the given string.

    Parameters:
    - string: String to be hashed.
    """
    return hashlib.md5(string.encode("utf-8")).hexdigest()


def save_overall_records(timestamp, description, done, save_dir, file_name):
    """
    Save or update records in an Excel file about experiments or tasks.

    Parameters:
    - timestamp: Timestamp of the record.
    - description: Description of the record.
    - done: Boolean indicating whether the task is completed.
    - save_dir: Directory where the records will be saved.
    - file_name: Base name of the file to save the records.
    """
    description = description.strip().replace(
        "\n", ";").replace(" ", "").replace("\t", "")
    df = pd.DataFrame([[timestamp, done, description]], columns=[
                      "TimeStamp", "done", "description"])
    save_path = os.path.join(save_dir, f"{file_name}_exp_records.xlsx")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_path):
        ori_df = pd.read_excel(save_path)
        if done:
            ind = ori_df[ori_df.TimeStamp == timestamp].index[0]
            ori_df.at[ind, 'done'] = done
            df = ori_df
        else:
            df = pd.concat([df, ori_df], ignore_index=True)

    df.to_excel(save_path, index=False)
    print(f"Saved overall records to {save_path}")
