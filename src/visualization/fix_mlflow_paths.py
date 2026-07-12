import os
import shutil
import sqlite3


def rewrite_mlflow_paths(db_path, new_artifact_root):
    """
    Updates the artifact_location in experiments and artifact_uri in runs
    from absolute paths to a container-agnostic path.
    """

    print(f"Backing up database to {db_path}.backup")
    shutil.copy2(db_path, f"{db_path}.backup")

    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 1. Update artifact_location in experiments table
        # We replace any absolute path ending in '/mlartifacts' with the new root
        print("Updating experiments table...")
        cursor.execute("SELECT experiment_id, artifact_location FROM experiments;")
        experiments = cursor.fetchall()

        for exp_id, location in experiments:
            if location and location.startswith("file://") and "/reports/mlartifacts" in location:
                # e.g. file:///home/user/BEHealSy/reports/mlartifacts/1
                # -> file:///mlartifacts/1
                new_location = location.split("/reports/mlartifacts")[1]
                new_location = f"{new_artifact_root}{new_location}"

                cursor.execute("UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?", (new_location, exp_id))
                print(f"  Exp {exp_id}: {location} -> {new_location}")

        # 2. Update artifact_uri in runs table
        print("\nUpdating runs table...")
        cursor.execute("SELECT run_uuid, artifact_uri FROM runs;")
        runs = cursor.fetchall()

        for run_uuid, uri in runs:
            if uri and uri.startswith("file://") and "/reports/mlartifacts" in uri:
                # e.g. file:///home/user/BEHealSy/reports/mlartifacts/1/run_id/artifacts
                # -> file:///mlartifacts/1/run_id/artifacts
                new_uri = uri.split("/reports/mlartifacts")[1]
                new_uri = f"{new_artifact_root}{new_uri}"

                cursor.execute("UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?", (new_uri, run_uuid))
                print(f"  Run {run_uuid}: {uri} -> {new_uri}")

        # Commit changes
        conn.commit()
        print("\nDatabase paths successfully updated!")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    # Path to your MLflow sqlite database
    DB_PATH = "./reports/mlflow.db"

    # The new root path that the Docker container expects
    NEW_ROOT = "file:///mlartifacts"

    rewrite_mlflow_paths(DB_PATH, NEW_ROOT)
