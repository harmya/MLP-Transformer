from manim import *
import math
import random
import numpy as np


class PositionalEncodingOne(Scene):
    def construct(self):
        positional_encoding = Text("Positional Encoding", font="Alliance No.1").scale(0.7).to_edge(UP)
        pe_formula_sin = Tex("$PE(pos, 2i) = sin(\\frac{\\text{pos}}{10000^{2i/d}})$")
        pe_formula_cos = Tex("$PE(pos, 2i + 1) = cos(\\frac{\\text{pos}}{10000^{2i/d}})$").next_to(pe_formula_sin, DOWN).align_to(pe_formula_sin, LEFT).shift(0.7*DOWN)
        pe_formulas = VGroup(pe_formula_sin, pe_formula_cos).scale(0.6).to_edge(LEFT)
        word = Text("PE at pos=1", font="Alliance No.1").scale(0.5).to_edge(UP).shift(0.2*DOWN).shift(1*DOWN)
        down_arrow = Arrow(word.get_bottom(), word.get_bottom() + 1 * DOWN, color=WHITE)
        matrix_elements = [[Tex("$PE(1, 1)$").scale(0.7)], [Tex("$PE(1, 2)$").scale(0.7)], [Tex("$PE(1, 3)$").scale(0.7)], [Tex("$PE(1, 4)$").scale(0.7)], [Text("...")], [Tex("$PE(1, d)$").scale(0.7)]]
        matrix = MobjectMatrix(matrix_elements, element_alignment_corner=DOWN).scale(0.7).next_to(down_arrow, DOWN)
        word_pe = VGroup(word, down_arrow, matrix).shift(0.8*LEFT)
        
        arrows_from_pe_sin = VGroup(
            Arrow(pe_formula_sin.get_right(), matrix_elements[0][0].get_left() + 0.15*RIGHT, color=RED, max_tip_length_to_length_ratio=0.06),
            Arrow(pe_formula_sin.get_right(), matrix_elements[2][0].get_left() + 0.15*RIGHT, color=RED, max_tip_length_to_length_ratio=0.06),
            Arrow(pe_formula_sin.get_right(), matrix_elements[4][0].get_left() + 0.15*RIGHT, color=RED, max_tip_length_to_length_ratio=0.06)
        )
        
        arrows_from_pe_cos = VGroup(
            Arrow(pe_formula_cos.get_right(), matrix_elements[1][0].get_left() + 0.15*RIGHT, color=GREEN, max_tip_length_to_length_ratio=0.06),
            Arrow(pe_formula_cos.get_right(), matrix_elements[3][0].get_left() + 0.15*RIGHT, color=GREEN, max_tip_length_to_length_ratio=0.06),
            Arrow(pe_formula_cos.get_right(), matrix_elements[5][0].get_left() + 0.15*RIGHT, color=GREEN, max_tip_length_to_length_ratio=0.06)
        )

        laughter = Text("laughter", font="Alliance No.1").scale(0.6).next_to(word, RIGHT).shift(0.5*RIGHT)
        down_arrow_laughter = Arrow(laughter.get_bottom(), laughter.get_bottom() + 1 * DOWN, color=WHITE)
        matrix_elements_laughter = [[Tex("$d_1$")], [Tex("$d_2$")], [Tex("$d_3$")], [Tex("$d_4$")], [Text("...")], [Tex("$d_d$")]]
        matrix_laughter = MobjectMatrix(matrix_elements_laughter, element_alignment_corner=DOWN).scale(0.7).next_to(down_arrow_laughter, DOWN)
        word_pe_laughter = VGroup(laughter, down_arrow_laughter, matrix_laughter)

        plus_sign = Tex("$+$").scale(1).move_to(
            (matrix.get_center() + matrix_laughter.get_center()) / 2
        ).shift(0.1*RIGHT)

        equals = Tex("$=$").scale(1).next_to(matrix_laughter, RIGHT)
        input_embeddings = Text("Input Embeddings", font="Alliance No.1").scale(0.5).next_to(laughter, RIGHT).shift(0.5*RIGHT)
        down_arrow_input_embeddings = Arrow(input_embeddings.get_bottom(), input_embeddings.get_bottom() + 1 * DOWN, color=WHITE)
        matrix_elements_final = [[Tex("$d_1 + PE(1, 1)$")], [Tex("$d_2 + PE(1, 2)$")], [Tex("$d_3 + PE(1, 3)$")], [Tex("$d_4 + PE(1, 4)$")], [Text("...")], [Tex("$d_d + PE(1, d)$")]]
        matrix_final = MobjectMatrix(matrix_elements_final, element_alignment_corner=DOWN).scale(0.7).next_to(equals, RIGHT)
        word_pe_add = VGroup(matrix, plus_sign) 
        word_pe_final = VGroup(input_embeddings, down_arrow_input_embeddings, matrix_final)

        self.play(Write(positional_encoding)) 
        self.wait(0.5)
        self.play(Write(pe_formulas))
        self.wait(0.5)
        self.play(Write(word_pe))
        self.wait(0.5)
        self.play(Write(arrows_from_pe_sin))
        self.wait(0.5)
        self.play(Write(arrows_from_pe_cos))
        self.wait(0.5)
        self.play(Write(word_pe_laughter))
        self.wait(0.5)
        self.play(Write(plus_sign))
        self.wait(0.5)
        self.play(Write(equals))
        self.wait(0.5)
        self.play(Write(word_pe_final))
        self.wait(2)

class word_embedding(Scene):
    def construct(self):
        sentence = "the laughter makes me happy"
        sentence_object = Text(sentence, font="Alliance No.1").scale(0.7).to_edge(UP).shift(0.2*DOWN).shift(0.2*RIGHT)
        
        words = sentence.split(" ")
        words_boxes = VGroup()
        random_ints = ["54", "23", "12", "98", "43"]
        random_decimals = [
            ["0.54", "0.23", "...", "0.98", "0.43"],
            ["0.92", "0.42", "...", "0.12", "0.23"],
            ["0.12", "0.32", "...", "0.23", "0.92"],
            ["0.23", "0.12", "...", "0.54", "0.43"],
            ["0.43", "0.98", "...", "0.23", "0.12"]
        ]
        for i, word in enumerate(words):
            text = Text(word, font="Alliance No.1").scale(0.5)
            box = Rectangle(width=1.5, height=0.75, color=BLUE, fill_opacity=0.3, fill_color=BLUE)
            text.move_to(box.get_center())
            d_arrow_one = Arrow(box.get_bottom(), box.get_bottom() + 1 * DOWN, color=WHITE).shift(0.2*UP)

            random_int = random_ints[i]
            number = Text(str(random_int)).scale(0.7)
            number_box = Rectangle(width=1, height=0.5, color=BLUE, fill_opacity=0.3, fill_color=BLUE).next_to(d_arrow_one, DOWN).shift(0.2*UP)
            number.move_to(number_box.get_center())
            d_arrow_two = Arrow(number_box.get_bottom(), number_box.get_bottom() + 1 * DOWN, color=WHITE).shift(0.2*UP)

            matrix_elements = [[Text(str(random_decimal)).scale(0.5)] for random_decimal in random_decimals[i]]
            matrix = MobjectMatrix(matrix_elements, element_alignment_corner=DOWN, v_buff=0.35, bracket_h_buff=0.1).next_to(d_arrow_two, DOWN).shift(0.2*UP)
            matrix_dimension = Tex("$d \\times 1$").scale(0.5).next_to(matrix, DOWN).shift((0.65 * matrix.width) * RIGHT).shift(0.17*UP)
            words_boxes.add(VGroup(text, box, d_arrow_one, number, number_box, d_arrow_two, matrix, matrix_dimension))
        
        words_boxes.arrange(RIGHT, buff=0.1).next_to(sentence_object, DOWN).shift(0.5*DOWN).shift(0.5*RIGHT)
        token = Text("Token", font="Alliance No.1").scale(0.5).next_to(words_boxes[0], LEFT).shift(0.75*UP)
        word_embedding = Text("Word Embedding", font="Alliance No.1").scale(0.5).next_to(words_boxes[0], LEFT).shift(1.2*DOWN).shift(0.2*RIGHT)
        
        self.play(Write(sentence_object))
        self.wait(0.5)
        self.play(Write(words_boxes), run_time=4)
        self.play(Write(token))
        self.play(Write(word_embedding))
        self.wait(2)

class attention(Scene):
    def construct(self):
        sentence = ["the", "laughter", "makes", "me", "happy"]
        
        word_1 = Text(sentence[0], font="Alliance No.1", color=RED).scale(0.7)
        word_2 = Text(sentence[1], font="Alliance No.1", color=BLUE).scale(0.7)
        word_3 = Text(sentence[2], font="Alliance No.1", color=GREEN).scale(0.7)
        word_4 = Text(sentence[3], font="Alliance No.1", color=ORANGE).scale(0.7)
        word_5 = Text(sentence[4], font="Alliance No.1", color=YELLOW).scale(0.7)

        sentence_object = VGroup(word_1, word_2, word_3, word_4, word_5).arrange(RIGHT, buff=0.2).to_edge(UP).shift(0.2*DOWN).shift(0.2*RIGHT).to_edge(LEFT).scale(0.75)

        down_arrow = Arrow(sentence_object.get_bottom(), sentence_object.get_bottom() + 1 * DOWN, color=WHITE)
        add_pos_enconding = Tex("$+ \\text{Positional Encoding}$").scale(0.5).next_to(down_arrow, RIGHT)
        matrix_elements = [[Text("the", font="Alliance No.1")], [Text("laughter", font="Alliance No.1")], [Text("makes", font="Alliance No.1")], [Text("me", font="Alliance No.1")], [Text("happy", font="Alliance No.1")]]
        matrix_words = MobjectMatrix(matrix_elements, element_alignment_corner=DOWN).scale(0.7).next_to(down_arrow, DOWN)
        matrix_embed = [
            [Tex("$d_{00} \\quad d_{01} \\quad d_{02} \\quad d_{03} \\cdots \\quad d_{0d}$",color=RED).scale(0.8).move_to(matrix_elements[0][0].get_center())],
            [Tex("$d_{10} \\quad d_{11} \\quad d_{12} \\quad d_{13} \\cdots \\quad d_{1d}$",color=BLUE).scale(0.8).move_to(matrix_elements[1][0].get_center())],
            [Tex("$d_{20} \\quad d_{21} \\quad d_{22} \\quad d_{23} \\cdots \\quad d_{2d}$",color=GREEN).scale(0.8).move_to(matrix_elements[2][0].get_center())],
            [Tex("$d_{30} \\quad d_{31} \\quad d_{32} \\quad d_{33} \\cdots \\quad d_{3d}$",color=ORANGE).scale(0.8).move_to(matrix_elements[3][0].get_center())],
            [Tex("$d_{40} \\quad d_{41} \\quad d_{42} \\quad d_{43} \\cdots \\quad d_{4d}$",color=YELLOW).scale(0.8).move_to(matrix_elements[4][0].get_center())]
        ]
        matrix_embed_notation = MobjectMatrix(matrix_embed).scale(0.7).next_to(down_arrow, DOWN)
        matrix_embed_dimension = Tex("$n \\times d$").scale(0.5).next_to(matrix_embed_notation, DOWN).shift((0.5 * matrix_embed_notation.width) * RIGHT + 0.2 * UP)
        matrix_embed_notation = VGroup(matrix_embed_notation, matrix_embed_dimension)


        x_equals = Tex("=").next_to(matrix_embed_notation, LEFT)
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

        x_times_wq = [
            [Tex("$d_{00} \\times h_{00} + d_{01} \\times h_{10} + d_{02} \\times h_{20} + \\cdots + d_{0d} \\times h_{d0}$").scale(0.7)],
            [Tex("$d_{10} \\times h_{00} + d_{11} \\times h_{10} + d_{12} \\times h_{20} + \\cdots + d_{1d} \\times h_{d0}$").scale(0.7)],
            [Tex("$d_{20} \\times h_{00} + d_{21} \\times h_{10} + d_{22} \\times h_{20} + \\cdots + d_{2d} \\times h_{d0}$").scale(0.7)],
            [Tex("$d_{30} \\times h_{00} + d_{31} \\times h_{10} + d_{32} \\times h_{20} + \\cdots + d_{3d} \\times h_{d0}$").scale(0.7)],
            [Tex("$d_{40} \\times h_{00} + d_{41} \\times h_{10} + d_{42} \\times h_{20} + \\cdots + d_{4d} \\times h_{d0}$").scale(0.7)]
        ]
        x_times_wq_matrix = MobjectMatrix(x_times_wq).scale(0.7).next_to(matrix_wq_equals, RIGHT).shift(2.2 * LEFT)
        x_times_wq_dimension = Tex("$n \\times 1$").scale(0.5).next_to(x_times_wq_matrix, DOWN).shift((0.5 * x_times_wq_matrix.width) * RIGHT)
        x_times_wq_group = VGroup(x_times_wq_matrix, x_times_wq_dimension)
        equals_Q = Tex("= Q").scale(0.7).next_to(x_times_wq_group, RIGHT)

        matrix_word_w = [
            [Tex("$w_{00} \cdot h_1$", color=RED).scale(0.7).move_to(x_times_wq[0][0].get_center())],
            [Tex("$w_{10} \cdot h_1$", color=BLUE).scale(0.7).move_to(matrix_elements[1][0].get_center())],
            [Tex("$w_{20} \cdot h_1$", color=GREEN).scale(0.7).move_to(matrix_elements[2][0].get_center())],
            [Tex("$w_{30} \cdot h_1$", color=ORANGE).scale(0.7).move_to(matrix_elements[3][0].get_center())],
            [Tex("$w_{40} \cdot h_1$", color=YELLOW).scale(0.7).move_to(matrix_elements[4][0].get_center())]
        ]
        matrix_word_w = MobjectMatrix(matrix_word_w).scale(0.7).next_to(matrix_wq_equals, RIGHT).shift(2.2 * LEFT)
        matrix_word_w_dimension = Tex("$n \\times 1$").scale(0.5).next_to(matrix_word_w, DOWN).shift((0.5 * matrix_word_w.width) * RIGHT + 0.2 * UP)
        matrix_word_w = VGroup(matrix_word_w, matrix_word_w_dimension)

        self.play(Write(sentence_object))
        self.play(Write(down_arrow))
        self.play(Write(add_pos_enconding))
        self.wait(0.5)
        self.play(Write(matrix_words))
        self.wait(0.5)
        self.wait(1)
        self.play(Write(x))
        self.play(Write(x_equals))
        self.wait(0.5)
        self.play(Write(wq_group), run_time=4)
        self.wait(1.5)
        self.play(FadeOut(wq_equals, matrix_wq, matrix_wq_dimension))
        self.play((x.animate.shift(0.5 * LEFT)))
        self.play((W_q.animate.next_to(x, RIGHT).shift(0.1* LEFT)))
        self.play((matrix_wq_expanded_group.animate.next_to(matrix_words, RIGHT).shift(0.2*DOWN)))
        self.play((matrix_wq_equals.animate.next_to(matrix_wq_expanded_group, RIGHT)).shift(0.2*LEFT))
        self.wait(1)
        self.play(Write(x_times_wq_group))
        self.wait(0.5)
        self.play(Transform(x_times_wq_group, matrix_word_w))
        self.play(x_times_wq_group.animate.shift(0.8*LEFT).shift(0.2*UP))
        self.play(Write(equals_Q))
        self.wait(2)
        
        
class self_attention(Scene):
    def construct(self):
        
        Q = Tex("$Q = x \\cdot W_q$").scale(0.7).to_edge(UP).to_edge(LEFT).shift(0.2*DOWN).shift(0.2*RIGHT)
        K = Tex("$K = x \\cdot W_k$").scale(0.7).next_to(Q, DOWN).shift(0.2*DOWN)
        V = Tex("$V = x \\cdot W_v$").scale(0.7).next_to(K, DOWN).shift(0.2*DOWN)

        Q_matrix_elements = [
            [Tex("$x^{(Q)}_{w_0}$", color=RED).scale(0.7)],
            [Tex("$x^{(Q)}_{w_1}$", color=BLUE).scale(0.7)],
            [Tex("$x^{(Q)}_{w_2}$", color=GREEN).scale(0.7)],
            [Tex("$x^{(Q)}_{w_3}$", color=ORANGE).scale(0.7)],
            [Tex("$x^{(Q)}_{w_4}$", color=YELLOW).scale(0.7)]
        ]
        Kt_matrix_elements = [
            [Tex("$x^{(K)}_{w_0}$", color=RED).scale(0.7), 
             Tex("$x^{(K)}_{w_1}$",  color=BLUE).scale(0.7), 
             Tex("$x^{(K)}_{w_2}$",  color=GREEN).scale(0.7), 
             Tex("$x^{(K)}_{w_3}$",  color=ORANGE).scale(0.7), 
             Tex("$x^{(K)}_{w_4}$",  color=YELLOW).scale(0.7)]
        ]

        


        Qkt_matrix = [
            [Tex("$x^{(Q)}_{w_0} \\cdot x^{(K)}_{w_0}$").scale(0.7), 
             Tex("$x^{(Q)}_{w_0} \\cdot x^{(K)}_{w_1}$").scale(0.7), 
             Tex("$x^{(Q)}_{w_0} \\cdot x^{(K)}_{w_2}$").scale(0.7), 
             Tex("$x^{(Q)}_{w_0} \\cdot x^{(K)}_{w_3}$").scale(0.7), 
             Tex("$x^{(Q)}_{w_0} \\cdot x^{(K)}_{w_4}$").scale(0.7)],

            [Tex("$x^{(Q)}_{w_1} \\cdot x^{(K)}_{w_0}$").scale(0.7),
            Tex("$x^{(Q)}_{w_1} \\cdot x^{(K)}_{w_1}$").scale(0.7),
            Tex("$x^{(Q)}_{w_1} \\cdot x^{(K)}_{w_2}$").scale(0.7),
            Tex("$x^{(Q)}_{w_1} \\cdot x^{(K)}_{w_3}$").scale(0.7),
            Tex("$x^{(Q)}_{w_1} \\cdot x^{(K)}_{w_4}$").scale(0.7)],

            [Tex("$x^{(Q)}_{w_2} \\cdot x^{(K)}_{w_0}$").scale(0.7),
            Tex("$x^{(Q)}_{w_2} \\cdot x^{(K)}_{w_1}$").scale(0.7),
            Tex("$x^{(Q)}_{w_2} \\cdot x^{(K)}_{w_2}$").scale(0.7),
            Tex("$x^{(Q)}_{w_2} \\cdot x^{(K)}_{w_3}$").scale(0.7),
            Tex("$x^{(Q)}_{w_2} \\cdot x^{(K)}_{w_4}$").scale(0.7)],

            [Tex("$x^{(Q)}_{w_3} \\cdot x^{(K)}_{w_0}$").scale(0.7),
            Tex("$x^{(Q)}_{w_3} \\cdot x^{(K)}_{w_1}$").scale(0.7),
            Tex("$x^{(Q)}_{w_3} \\cdot x^{(K)}_{w_2}$").scale(0.7),
            Tex("$x^{(Q)}_{w_3} \\cdot x^{(K)}_{w_3}$").scale(0.7),
            Tex("$x^{(Q)}_{w_3} \\cdot x^{(K)}_{w_4}$").scale(0.7)],

            [Tex("$x^{(Q)}_{w_4} \\cdot x^{(K)}_{w_0}$").scale(0.7),
            Tex("$x^{(Q)}_{w_4} \\cdot x^{(K)}_{w_1}$").scale(0.7),
            Tex("$x^{(Q)}_{w_4} \\cdot x^{(K)}_{w_2}$").scale(0.7),
            Tex("$x^{(Q)}_{w_4} \\cdot x^{(K)}_{w_3}$").scale(0.7),
            Tex("$x^{(Q)}_{w_4} \\cdot x^{(K)}_{w_4}$").scale(0.7)]
        ]

        
        Qkt = Tex("$\\frac{Q \\cdot K^T}{\\sqrt{d}}$").scale(0.7).next_to(V, DOWN).to_edge(LEFT).shift(2*DOWN)
        A = Text("A", font="Alliance No.1").scale(0.7).next_to(V, DOWN).to_edge(LEFT).shift(2.1*DOWN)
        equals = Text("=").scale(0.7).next_to(Qkt, RIGHT)
        Qkt_matrix = MobjectMatrix(Qkt_matrix, h_buff=2).scale(0.7).next_to(equals, RIGHT)
        Qkt_dimension = Tex("$n \\times n$").scale(0.5).next_to(Qkt_matrix, DOWN).shift((0.5 * Qkt_matrix.width) * RIGHT + 0.1 * UP)

        V_matrix_elements = [
            [Tex("$x^{(V)}_{w_0}$", color=RED).scale(0.7)],
            [Tex("$x^{(V)}_{w_1}$", color=BLUE).scale(0.7)],
            [Tex("$x^{(V)}_{w_2}$", color=GREEN).scale(0.7)],
            [Tex("$x^{(V)}_{w_3}$", color=ORANGE).scale(0.7)],
            [Tex("$x^{(V)}_{w_4}$", color=YELLOW).scale(0.7)]
        ]
        V_matrix = MobjectMatrix(V_matrix_elements, element_alignment_corner=DOWN).scale(0.7).next_to(Qkt_matrix, RIGHT)
        V_matrix_dimension = Tex("$n \\times d$").scale(0.5).next_to(V_matrix, DOWN).shift((0.5 * V_matrix.width) * RIGHT + 0.2 * UP)

        self.play(Write(Q))
        self.wait(0.5)
        self.play(Write(K))
        self.wait(0.5)
        self.play(Write(V))
        self.wait(0.5)
        self.play(Write(Qkt))
        self.play(Write(equals))
        self.play(Write(Qkt_matrix))
        self.play(Write(Qkt_dimension))
        self.wait(0.5)
        self.play(Transform(Qkt, A))
        self.play(Write(V_matrix))
        self.play(Write(V_matrix_dimension))
        self.wait(2)


class multi_head_attention(Scene):
    def construct(self):

        x = Tex("$x$").scale(0.5).to_edge(LEFT).to_edge(UP).shift(1*DOWN).shift(0.2*RIGHT)
        W_q = Tex("$W_q^{(i)}$", color=RED).scale(0.5).next_to(x, RIGHT).shift(0.1 * LEFT)
        wq_equals = Tex("=").next_to(W_q, RIGHT)
        matrix_elements = [[Text("the", font="Alliance No.1")], [Text("laughter", font="Alliance No.1")], [Text("makes", font="Alliance No.1")], [Text("me", font="Alliance No.1")], [Text("happy", font="Alliance No.1")]]
        matrix_words = MobjectMatrix(matrix_elements, element_alignment_corner=DOWN).scale(0.3).next_to(wq_equals, RIGHT)
        matrix_wq_expanded_elements = [
            [Tex("$h_{00}$")],
            [Tex("$h_{10}$")],
            [Tex("$h_{20}$")],
            [Text("...")],
            [Tex("$h_{d0}$")],
        ]
        matrix_wq_expanded = MobjectMatrix(matrix_wq_expanded_elements, h_buff=1).scale(0.3).next_to(matrix_words, RIGHT)
        matrix_wq_expanded_dimension = Tex("$d \\times 1$").scale(0.3).next_to(matrix_wq_expanded, DOWN).shift((0.5 * matrix_wq_expanded.width) * RIGHT).shift(0.3*UP)
        matrix_wq_expanded_group = VGroup(matrix_wq_expanded, matrix_wq_expanded_dimension)

        whole = VGroup(x, W_q, wq_equals, matrix_words, matrix_wq_expanded_group)

        # duplicate
        whole_1 = whole.copy().shift(1.5*DOWN)
        whole_2 = whole.copy().shift(3*DOWN)
        whole_3 = whole.copy().shift(4.5*DOWN)
        
        self.play(Write(whole))
        self.wait(0.5)
        self.play(Write(whole_1))
        self.wait(0.5)
        self.play(Write(whole_2))
        self.wait(0.5)
        self.play(Write(whole_3))


        

        

        

        

        


