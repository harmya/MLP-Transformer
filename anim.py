from manim import *
import math
import random
import numpy as np

class attention(Scene):
    def construct(self):
        # show text

        sentence = "the laughter makes me happy"
        sentence_object = Text(sentence).scale(0.7).to_edge(UP).shift(0.2*DOWN)
        
        words = sentence.split(" ")
        words_boxes = VGroup()
        random_ints = ["54", "23", "12", "98", "43"]
        random_decimals = [
            ["0.54", "0.23", "0.12", "0.98", "0.43"],
            ["0.92", "0.42", "0.32", "0.12", "0.23"],
            ["0.12", "0.32", "0.42", "0.23", "0.92"],
            ["0.23", "0.12", "0.98", "0.54", "0.43"],
            ["0.43", "0.98", "0.54", "0.23", "0.12"]
        ]
        for i, word in enumerate(words):
            text = Text(word).scale(0.7)
            box = Rectangle(width=2, height=1, color=BLUE, fill_opacity=0.3, fill_color=BLUE)
            text.move_to(box.get_center())
            d_arrow_one = Arrow(box.get_bottom(), box.get_bottom() + 1 * DOWN, color=WHITE)

            random_int = random_ints[i]
            number = Text(str(random_int)).scale(0.7)
            number_box = Rectangle(width=1, height=0.5, color=BLUE, fill_opacity=0.3, fill_color=BLUE).next_to(d_arrow_one, DOWN)
            number.move_to(number_box.get_center())
            d_arrow_two = Arrow(number_box.get_bottom(), number_box.get_bottom() + 1 * DOWN, color=WHITE)

            matrix_elements = [[Text(str(random_decimal))] for random_decimal in random_decimals[i]]
            matrix = MobjectMatrix(matrix_elements, element_alignment_corner=DOWN).scale(0.5).next_to(d_arrow_two, DOWN)
    
            words_boxes.add(VGroup(text, box, d_arrow_one, number, number_box, d_arrow_two, matrix))
        
        words_boxes.arrange(RIGHT, buff=0.1).next_to(sentence_object, DOWN)

        self.play(Write(sentence_object))
        self.wait(0.5)
        self.play(Write(words_boxes), run_time=4)
        self.wait(0.5)
        self.play(FadeOut(words_boxes))
        self.wait(0.5)
        self.play(sentence_object.animate.to_edge(LEFT).to_edge(UP).shift(0.2*DOWN).shift(0.2*RIGHT))

        down_arrow = Arrow(sentence_object.get_bottom(), sentence_object.get_bottom() + 1 * DOWN, color=WHITE)
        matrix_elements = [[Text("the")], [Text("laughter")], [Text("makes")], [Text("me")], [Text("happy")]]
        matrix_words = MobjectMatrix(matrix_elements, element_alignment_corner=DOWN).scale(0.7).next_to(down_arrow, DOWN)
        matrix_elements_notation = [[Tex("$w_1$")], [Tex("$w_2$")], [Tex("$w_3$")], [Tex("$w_4$")], [Tex("$w_5$")]]
        matrix_word_notation = MobjectMatrix(matrix_elements_notation).scale(0.7).next_to(down_arrow, DOWN)
        matrix_word_notation_dimension = Tex("$5 \\times d$").scale(0.5).next_to(matrix_word_notation, DOWN).shift((0.5 * matrix_word_notation.width) * RIGHT)
        matrix_word_notation = VGroup(matrix_word_notation, matrix_word_notation_dimension)
        help_text = Tex("where $w_n$ is a word embedding of dimension $d$").scale(0.7).to_edge(DOWN).shift(0.1*UP).to_edge(LEFT).shift(0.1*RIGHT)       
    
        x_equals = Tex("=").next_to(matrix_word_notation, LEFT)
        x = Tex("$x$").scale(0.7).next_to(x_equals, LEFT)

        W_q = Tex("$W_q$", color=RED).scale(0.7).next_to(x, RIGHT).shift(6 * RIGHT)
        wq_equals = Tex("=").next_to(W_q, RIGHT)

        matrix_wq_elements = [[Tex("$h_1$")]]   
        matrix_wq = MobjectMatrix(matrix_wq_elements, h_buff=1).scale(0.7).next_to(wq_equals, RIGHT)
        matrix_wq_dimension = Tex("$d \\times 1$").scale(0.5).next_to(matrix_wq, DOWN).shift((0.5 * matrix_wq.width) * RIGHT + 0.2 * UP)
        matrix_wq = VGroup(matrix_wq, matrix_wq_dimension)
        matrix_wq_group = VGroup(W_q, wq_equals, matrix_wq, matrix_wq_dimension)
        matrix_wq_equals = Tex("=").next_to(matrix_wq, RIGHT)
        matrix_wq_expanded_elements = [
            [Tex("$h_{00}$")],
            [Tex("$h_{10}$")],
            [Tex("$h_{20}$")],
            [Text("...")],
            [Tex("$h_{d0}$")],
        ]
        matrix_wq_expanded = MobjectMatrix(matrix_wq_expanded_elements, h_buff=1).scale(0.7).next_to(matrix_wq_equals, RIGHT)
        matrix_wq_expanded_dimension = Tex("$d \\times 1$").scale(0.5).next_to(matrix_wq_expanded, DOWN).shift((0.5 * matrix_wq_expanded.width) * RIGHT)
        matrix_wq_expanded_group = VGroup(matrix_wq_expanded, matrix_wq_expanded_dimension)
        wq_group = VGroup(W_q, wq_equals, matrix_wq_group, matrix_wq_equals, matrix_wq_expanded_group)


        self.play(Write(down_arrow))
        self.wait(0.5)
        self.play(Write(matrix_words))
        self.wait(0.5)
        self.play(Transform(matrix_words, matrix_word_notation))
        self.play(Write(x))
        self.play(Write(x_equals))
        self.play(Write(help_text))
        self.wait(0.5)
        self.play(Write(wq_group), run_time=4)
        self.wait(0.5)
        self.play((x.animate.shift(1.5 * LEFT)))
        self.play((W_q.animate.next_to(x, RIGHT).shift(0.15*LEFT)))
        self.play((wq_equals.animate.next_to(W_q, RIGHT)))
        self.play((matrix_words.animate.next_to(wq_equals, RIGHT)))
        self.play((matrix_wq.animate.next_to(matrix_words, RIGHT).shift(0.2*LEFT)))
    
        

        


        

        


