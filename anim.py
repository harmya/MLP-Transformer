from manim import *

class attention(Scene):
    def construct(self):
        # show text

        # Sentence
        sentence = Text("the laughter makes me happy").scale(0.7).to_edge(UP).to_edge(LEFT).shift(0.2*RIGHT).shift(0.2*DOWN)
        down_arrow = Tex(r"$\downarrow$").scale(2).next_to(sentence, DOWN)
        sentence = VGroup(sentence, down_arrow)

        matrix_elements = [[Text("the")], [Text("laughter")], [Text("makes")], [Text("me")], [Text("happy")]]
        matrix_words = MobjectMatrix(matrix_elements, element_alignment_corner=DOWN).scale(0.7).next_to(down_arrow, DOWN)

        matrix_elements_notation = [[Tex("$w_1$")], [Tex("$w_2$")], [Tex("$w_3$")], [Tex("$w_4$")], [Tex("$w_5$")]]
        matrix_word_notation = MobjectMatrix(matrix_elements_notation).scale(0.7).next_to(down_arrow, DOWN)
        matrix_word_notation_dimension = Tex("$5 \\times d$").scale(0.5).next_to(matrix_word_notation, DOWN).shift((0.5 * matrix_word_notation.width) * RIGHT)
        matrix_word_notation = VGroup(matrix_word_notation, matrix_word_notation_dimension)

        help_text = Tex("where $w_n$ is a word embedding of dimension $d$").scale(0.7).to_edge(DOWN).shift(0.1*UP).to_edge(LEFT).shift(0.1*RIGHT)       
        
        x_equals = Tex("=").next_to(matrix_word_notation, LEFT)
        x = Tex("$x$").scale(0.7).next_to(x_equals, LEFT)

        W_q = Tex("$W_q$", color=RED).scale(0.7)
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


        self.play(Write(sentence))
        self.wait(0.5)
        self.play(Write(matrix_words))
        self.wait(0.5)
        self.play(Transform(matrix_words, matrix_word_notation))
        self.play(Write(x))
        self.play(Write(x_equals))
        self.play(Write(help_text))
        self.wait(0.5)
        # self.play(FadeOut(sentence), FadeOut(down_arrow), FadeOut(matrix_words), FadeOut(matrix_word_notation))
        self.wait(0.5)
        self.play(Write(wq_group), run_time=4)
        self.wait(0.5)
        self.play(FadeOut(sentence), FadeOut(down_arrow), FadeOut(x_equals), FadeOut(matrix_wq_equals), FadeOut(matrix_wq_expanded_group))
        self.play((x.animate.shift(1.5 * LEFT)))
        self.play((W_q.animate.next_to(x, RIGHT).shift(0.15*LEFT)))
        self.play((wq_equals.animate.next_to(W_q, RIGHT)))
        self.play((matrix_words.animate.next_to(wq_equals, RIGHT)))
        self.play((matrix_wq.animate.next_to(matrix_words, RIGHT).shift(0.2*LEFT)))
        

        


        

        


