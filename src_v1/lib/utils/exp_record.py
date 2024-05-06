import logging
import datetime
import json
import argparse
import yaml
import hashlib

import os
import pandas as pd
import wandb
import git


def git_code(commit_info=""):
    repo_dir = os.getcwd()  # 当前目录
    repo = git.Repo(repo_dir)

    repo.git.add(".")  # 添加所有文件到缓存区
    repo.git.commit("-m", commit_info)  # 提交代码并添加提交信息
    print("success git current version")

    # get id
    commit = repo.head.commit
    commit_id = commit.hexsha

    return commit_id


def wandb_init(job_type, id, name, config, project, notes, wandb_dir):
    run = wandb.init(
        job_type=job_type,
        id=id,
        name=name,
        project=project,
        config=config,
        notes=notes,
        dir=wandb_dir,
        save_code=True,
    )
    return run


def init_logging(save_dir):
    # if not os.path.exists("log"):
    #     os.makedirs("log", exist_ok=True)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    filename = os.path.join(save_dir, "log.log")
    logging.basicConfig(
        filename=filename,
        format="%(asctime)s - %(pathname)s[line:%(lineno)d]: %(message)s",
        level=logging.INFO,
    )

    return filename


def params_to_yaml(args, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "exp_args.yaml")

    # Save the parser object to a YAML file
    with open(save_path, "w") as outfile:
        yaml.dump(args, outfile)


def hash_string(string):
    md = hashlib.md5(string.encode("utf-8"))

    return md.hexdigest()


def save_overall_records(timestamp, description, done, save_dir, file_name):
    # now = datetime.datetime.now()

    description = (
        description.strip("\n")
        .replace("\n", ";")
        .replace(" ", "")
        .replace("\t", "")
        .replace(";", "")
    )

    df = pd.DataFrame(
        [[timestamp, done, description]], columns=["TimeStamp", "done", "description"]
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "%s_exp_records.xlsx" % file_name)
    # save_path = os.path.join(save_dir, "%s_exp_records.csv" % file_name)
    if os.path.exists(save_path):
        if not done:
            ori_df = pd.read_excel(save_path)
            # ori_df = pd.read_csv(save_path, index_col=0)
            # df = df.append(ori_df)
            df = pd.concat([df, ori_df])
        else:
            df = pd.read_excel(save_path)
            # df = pd.read_csv(save_path)
            ind = df[df.TimeStamp == timestamp].index.tolist()[0]
            df.done.iloc[ind] = done

    df.to_excel(save_path, index=False, index_label=None)
    # df.to_csv(save_path, index=False, index_label=None)

    print("save overall records to %s" % save_path)
