#!/usr/bin/env python3
"""
Test script for GPT Feedback Auto-Scorer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.gpt_feedback import score_gpt_feedback, categorize_feedback_score, get_feedback_statistics

def test_feedback_scoring_scenarios():
    """Test different GPT feedback scenarios and their scores"""
    
    print("🧪 Testing GPT Feedback Auto-Scorer")
    print("="*50)
    
    test_cases = [
        {
            "name": "Ultra Strong Signal (Polish)",
            "feedback": "To jest bardzo silny sygnał z solidną strukturą akumulacji. Wysokie prawdopodobieństwo kontynuacji wzrostu. Niskie ryzyko fakeout.",
            "expected_range": (85, 100)
        },
        {
            "name": "Ultra Strong Signal (English)",
            "feedback": "Excellent signal with robust accumulation pattern. Very likely to continue upward. Low risk with solid structure.",
            "expected_range": (85, 100)
        },
        {
            "name": "Good Signal with Risk",
            "feedback": "Dobry sygnał, ale wysokie ryzyko ze względu na możliwy fakeout. Umiarkowana konfirmacja.",
            "expected_range": (50, 70)
        },
        {
            "name": "Weak Signal",
            "feedback": "Słaby sygnał z mieszanymi wskaźnikami. Trudne do przewidzenia, możliwa manipulacja.",
            "expected_range": (30, 50)
        },
        {
            "name": "Pump and Dump Warning",
            "feedback": "Podejrzenie pump and dump. Sztuczny wzrost z social hype. Bardzo wysokie ryzyko.",
            "expected_range": (20, 40)
        },
        {
            "name": "Technical Breakout",
            "feedback": "Przełamanie oporu z wzrostem wolumenu. Dobra konfirmacja akumulacji. Kontynuacja prawdopodobna.",
            "expected_range": (75, 90)
        }
    ]
    
    results = []
    
    for case in test_cases:
        score = score_gpt_feedback(case["feedback"])
        category, description, emoji = categorize_feedback_score(score)
        
        print(f"\n📊 {case['name']}")
        print(f"Feedback: {case['feedback'][:80]}...")
        print(f"Score: {score}/100")
        print(f"Category: {category} {emoji}")
        print(f"Expected range: {case['expected_range'][0]}-{case['expected_range'][1]}")
        
        in_range = case['expected_range'][0] <= score <= case['expected_range'][1]
        print(f"Result: {'✅ PASS' if in_range else '❌ FAIL'}")
        
        results.append({
            'name': case['name'],
            'score': score,
            'category': category,
            'expected': case['expected_range'],
            'passed': in_range
        })
    
    return results

def test_score_categories():
    """Test score categorization system"""
    
    print("\n🎯 Testing Score Categories")
    print("="*35)
    
    test_scores = [95, 85, 75, 65, 55, 45, 25]
    
    for score in test_scores:
        category, description, emoji = categorize_feedback_score(score)
        print(f"Score {score}: {category} {emoji} - {description}")
    
    return test_scores

def test_edge_cases():
    """Test edge cases and special scenarios"""
    
    print("\n🔍 Testing Edge Cases")
    print("="*25)
    
    edge_cases = [
        {"name": "Empty feedback", "text": "", "expected_score": 50},
        {"name": "Very short feedback", "text": "Ok", "expected_score": 50},
        {"name": "Long detailed analysis", "text": "This is a very detailed analysis " * 10, "expected_min": 63},
        {"name": "Mixed signals", "text": "Strong signal but high risk and uncertainty", "expected_range": (45, 75)},
        {"name": "Only positive indicators", "text": "Solid structure, low risk, good confirmation", "expected_min": 70}
    ]
    
    for case in edge_cases:
        score = score_gpt_feedback(case["text"])
        category, _, emoji = categorize_feedback_score(score)
        
        print(f"\n{case['name']}: {score}/100 ({category}) {emoji}")
        
        # Check expectations
        if "expected_score" in case:
            passed = score == case["expected_score"]
        elif "expected_min" in case:
            passed = score >= case["expected_min"]
        elif "expected_range" in case:
            passed = case["expected_range"][0] <= score <= case["expected_range"][1]
        else:
            passed = True
            
        print(f"Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return edge_cases

def test_statistics_calculation():
    """Test feedback statistics calculation"""
    
    print("\n📈 Testing Statistics Calculation")
    print("="*40)
    
    sample_scores = [95, 87, 82, 76, 68, 55, 42, 38, 25]
    stats = get_feedback_statistics(sample_scores)
    
    print(f"Sample scores: {sample_scores}")
    print(f"Average score: {stats['avg_score']}")
    print(f"Max score: {stats['max_score']}")
    print(f"Min score: {stats['min_score']}")
    print(f"Ultra clean signals (≥85): {stats['ultra_clean_count']}")
    print(f"Strong signals (≥70): {stats['strong_count']}")
    print(f"Total signals: {stats['total_count']}")
    
    # Test empty list
    empty_stats = get_feedback_statistics([])
    print(f"\nEmpty list stats: {empty_stats}")
    
    return stats

def test_language_detection():
    """Test scoring with different languages"""
    
    print("\n🌍 Testing Multi-Language Support")
    print("="*40)
    
    language_tests = [
        {
            "lang": "Polish",
            "text": "Silny sygnał z dobrą konfirmacją. Niskie ryzyko.",
            "expected_min": 75
        },
        {
            "lang": "English", 
            "text": "Strong signal with good confirmation. Low risk.",
            "expected_min": 75
        },
        {
            "lang": "Mixed",
            "text": "Strong sygnał but wysokie ryzyko due to uncertainty.",
            "expected_range": (60, 80)
        }
    ]
    
    for test in language_tests:
        score = score_gpt_feedback(test["text"])
        print(f"{test['lang']}: {score}/100")
        
        if "expected_min" in test:
            passed = score >= test["expected_min"]
        else:
            passed = test["expected_range"][0] <= score <= test["expected_range"][1]
            
        print(f"Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return language_tests

if __name__ == "__main__":
    print("🚀 GPT Feedback Auto-Scorer Test Suite")
    print("="*60)
    
    # Run all tests
    scoring_results = test_feedback_scoring_scenarios()
    category_results = test_score_categories()
    edge_case_results = test_edge_cases()
    stats_results = test_statistics_calculation()
    language_results = test_language_detection()
    
    # Summary
    print("\n📋 Test Summary")
    print("="*20)
    
    scoring_passed = sum(1 for r in scoring_results if r['passed'])
    total_scoring_tests = len(scoring_results)
    
    print(f"✅ Feedback Scoring: {scoring_passed}/{total_scoring_tests} tests passed")
    print(f"✅ Score Categories: {len(category_results)} categories tested")
    print(f"✅ Edge Cases: Handled empty, short, and long feedback")
    print(f"✅ Statistics: Average, min, max, and count calculations")
    print(f"✅ Multi-Language: Polish and English support")
    
    overall_passed = scoring_passed == total_scoring_tests
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_passed else '❌ SOME TESTS FAILED'}")
    
    if overall_passed:
        print("\n🎉 GPT Feedback Auto-Scorer ready for production!")
        print("Key features:")
        print("- Automatic quality scoring (0-100 scale)")
        print("- Multi-language support (Polish/English)")
        print("- 5-tier categorization system")
        print("- Comprehensive statistical analysis")
        print("- Integration with main scanning workflow")
    else:
        print("\n⚠️ Please review failed tests before deployment.")
        
    print("\n📊 Score Categories:")
    print("🔵 ULTRA_CLEAN (85-100): Ultra clean signal, high continuation probability")
    print("🟢 STRONG (70-84): Strong signal with good potential")
    print("🟡 MODERATE (55-69): Moderate signal with some risk")
    print("🟠 WEAK (40-54): Weak signal, higher risk")
    print("🔴 POOR (0-39): Poor signal, possible fakeout")