import git


def get_git_root(path):
    """ returns the root folder of a git repository given a path """
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root