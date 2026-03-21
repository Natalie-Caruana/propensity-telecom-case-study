import 'tasks/check.just'
import 'tasks/install.just'
import 'tasks/clean.just'
import 'tasks/train.just'

# Show available recipes
default:
    @just --list
