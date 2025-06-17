import json
from collections import defaultdict
import argparse

def analyze_benchmark_results(baseline_file, post_training_file=None):
    """Analyze and compare benchmark results"""
    
    print("📊 SUDOKU BENCHMARK ANALYSIS")
    print("=" * 50)
    
    # Load baseline results
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    post_training = None
    if post_training_file:
        with open(post_training_file, 'r') as f:
            post_training = json.load(f)
    
    def analyze_dataset(results, name):
        """Analyze a single dataset"""
        total_puzzles = len(results['results'])
        perfect_solutions = sum(1 for r in results['results'] if r.get('success', False))
        avg_accuracy = sum(r.get('evaluation', {}).get('accuracy', 0) for r in results['results']) / total_puzzles if total_puzzles > 0 else 0
        avg_coverage = sum(r.get('evaluation', {}).get('coverage', 0) for r in results['results']) / total_puzzles if total_puzzles > 0 else 0
        avg_precision = sum(r.get('evaluation', {}).get('precision', 0) for r in results['results']) / total_puzzles if total_puzzles > 0 else 0
        
        print(f"\n{name} Results:")
        print(f"  Total puzzles: {total_puzzles}")
        print(f"  Perfect solutions: {perfect_solutions}/{total_puzzles} ({perfect_solutions/total_puzzles*100:.1f}%)")
        print(f"  Average accuracy: {avg_accuracy:.3f}")
        print(f"  Average coverage: {avg_coverage:.3f}")
        print(f"  Average precision: {avg_precision:.3f}")
        
        # Analyze by difficulty
        by_difficulty = defaultdict(list)
        for result in results['results']:
            difficulty = result.get('difficulty', 'unknown')
            by_difficulty[difficulty].append(result)
        
        print(f"  \n  By Difficulty:")
        for difficulty in ['beginner', 'easy', 'medium', 'hard', 'expert']:
            if difficulty in by_difficulty:
                diff_results = by_difficulty[difficulty]
                perfect = sum(1 for r in diff_results if r.get('success', False))
                avg_acc = sum(r.get('evaluation', {}).get('accuracy', 0) for r in diff_results) / len(diff_results)
                print(f"    {difficulty.capitalize()}: {perfect}/{len(diff_results)} perfect ({perfect/len(diff_results)*100:.1f}%), avg accuracy: {avg_acc:.3f}")
        
        return {
            'total': total_puzzles,
            'perfect': perfect_solutions,
            'accuracy': avg_accuracy,
            'coverage': avg_coverage,
            'precision': avg_precision,
            'by_difficulty': by_difficulty
        }
    
    # Analyze baseline
    baseline_stats = analyze_dataset(baseline, "BASELINE")
    
    # Analyze post-training if available
    if post_training:
        post_stats = analyze_dataset(post_training, "POST-TRAINING")
        
        # Compare results
        print(f"\n🔄 IMPROVEMENT ANALYSIS")
        print("=" * 50)
        
        perfect_improvement = post_stats['perfect'] - baseline_stats['perfect']
        accuracy_improvement = post_stats['accuracy'] - baseline_stats['accuracy']
        
        print(f"Perfect solutions: {baseline_stats['perfect']} → {post_stats['perfect']} ({perfect_improvement:+d})")
        print(f"Average accuracy: {baseline_stats['accuracy']:.3f} → {post_stats['accuracy']:.3f} ({accuracy_improvement:+.3f})")
        
        # Difficulty-wise comparison
        print(f"\nBy Difficulty:")
        for difficulty in ['beginner', 'easy', 'medium', 'hard', 'expert']:
            if difficulty in baseline_stats['by_difficulty'] and difficulty in post_stats['by_difficulty']:
                baseline_diff = baseline_stats['by_difficulty'][difficulty]
                post_diff = post_stats['by_difficulty'][difficulty]
                
                baseline_acc = sum(r.get('evaluation', {}).get('accuracy', 0) for r in baseline_diff) / len(baseline_diff)
                post_acc = sum(r.get('evaluation', {}).get('accuracy', 0) for r in post_diff) / len(post_diff)
                
                baseline_perfect = sum(1 for r in baseline_diff if r.get('success', False))
                post_perfect = sum(1 for r in post_diff if r.get('success', False))
                
                acc_improvement = post_acc - baseline_acc
                perfect_improvement = post_perfect - baseline_perfect
                
                print(f"  {difficulty.capitalize()}:")
                print(f"    Accuracy: {baseline_acc:.3f} → {post_acc:.3f} ({acc_improvement:+.3f})")
                print(f"    Perfect: {baseline_perfect} → {post_perfect} ({perfect_improvement:+d})")
        
        # Overall assessment
        print(f"\n🎯 TRAINING ASSESSMENT")
        print("=" * 50)
        
        if accuracy_improvement > 0.1:
            print("✅ EXCELLENT: Significant improvement in accuracy!")
        elif accuracy_improvement > 0.05:
            print("✅ GOOD: Noticeable improvement in accuracy")
        elif accuracy_improvement > 0.01:
            print("🔶 MODEST: Small improvement in accuracy")
        elif accuracy_improvement > -0.01:
            print("🔶 NEUTRAL: No significant change")
        else:
            print("❌ REGRESSION: Model performance decreased")
        
        if perfect_improvement > 0:
            print(f"✅ More perfect solutions achieved (+{perfect_improvement})")
        elif perfect_improvement == 0:
            print("🔶 Same number of perfect solutions")
        else:
            print(f"❌ Fewer perfect solutions ({perfect_improvement})")

def main():
    parser = argparse.ArgumentParser(description="Analyze Sudoku benchmark results")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline results file")
    parser.add_argument("--post_training", type=str, help="Post-training results file")
    
    args = parser.parse_args()
    
    analyze_benchmark_results(args.baseline, args.post_training)

if __name__ == "__main__":
    main()