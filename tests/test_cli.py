from relu_tm.cli import main


def test_list(capsys):
    assert main(["list"]) == 0
    assert "bb4" in capsys.readouterr().out


def test_run_example(capsys):
    assert main(["run", "bb2"]) == 0
    assert "All 6 steps agree" in capsys.readouterr().out


def test_run_standard_format_and_input(capsys):
    assert main(["run", "1RB1LB_1LA1RZ", "-i", "11"]) == 0
    assert main(["run", "palindrome", "-i", "abba", "--every", "5"]) == 0


def test_run_float32_failure_exit_code(capsys):
    assert main(["run", "filler", "-n", "30", "-b", "float32"]) == 1
    assert "First disagreement" in capsys.readouterr().out


def test_non_halting_needs_steps(capsys):
    assert main(["run", "counter"]) == 2


def test_info_with_program(capsys):
    assert main(["info", "bb3", "--program"]) == 0
    out = capsys.readouterr().out
    assert "width 4" in out and "case 0" in out
